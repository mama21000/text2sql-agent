"""
Text-to-SQL agent with an error-class-conditioned repair loop.

The idea: when a generated query fails, the *kind* of failure tells you what to
do about it. A syntax error needs the syntax fixed while the logic is kept. An
unknown column means the schema was misread. An empty result usually means the
filter value was guessed wrong, and the fix is to go look at the actual data.
Treating all three identically - which is what blind retry does - wastes the
most informative signal available.

Graph shape:

    generate -> execute -> assess -> [routing decision]
                                       |- done      -> END
                                       |- give_up   -> END
                                       '- otherwise -> investigate -> generate

`investigate` gathers whatever extra evidence the error class calls for, then
hands control back to `generate` with that evidence in state.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "4"))


# ---------------------------------------------------------------------------
# 0. Rate-limit handling
# ---------------------------------------------------------------------------

async def with_retry(coro_fn, attempts: int = 5):
    """
    Retry a model call on rate limits with exponential backoff.

    Free tiers throttle aggressively, and a long evaluation run will hit the
    limit repeatedly. Without this, one 429 kills the whole run.
    """
    import asyncio
    import random

    for i in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:
            message = str(exc).lower()
            rate_limited = any(s in message for s in
                               ("429", "resource_exhausted", "rate limit",
                                "quota", "too many requests"))
            if not rate_limited or i == attempts - 1:
                raise
            wait = min(60.0, 5.0 * (2 ** i)) + random.uniform(0, 2)
            print(f"    rate limited, waiting {wait:.0f}s...", flush=True)
            await asyncio.sleep(wait)


def _is_structured_output_failure(exc: Exception) -> bool:
    """
    Detect the provider rejecting its own tool call.

    Some providers wrap structured output in function calling and then fail to
    parse the model's arguments as JSON. SQL string literals are the usual
    trigger: a query containing 'S' comes back escaped as \\'S\\', which is
    valid Python and invalid JSON, so the whole call is rejected even though
    the SQL itself was fine. Falling back to plain text avoids the wrapper.
    """
    message = str(exc).lower()
    return any(s in message for s in
               ("tool_use_failed", "failed to call a function",
                "failed_generation", "invalid_request_error"))


# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------

def build_llm(temperature: float = 0.0):
    """
    Gemini by default (generous free tier, 1M context so schemas fit).
    Set LLM_PROVIDER=groq to switch.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 2. Structured outputs
# ---------------------------------------------------------------------------

class GeneratedSQL(BaseModel):
    """What the generate node must return."""
    sql: str = Field(description="A single SQLite SELECT statement, no markdown fences.")
    reasoning: str = Field(description="One sentence on the approach taken.")


class Assessment(BaseModel):
    """
    The assess node's verdict. `next_action` is what the conditional edge routes
    on, so it must be one of the known branch names.
    """
    next_action: Literal[
        "done",
        "fix_syntax",
        "fix_schema",
        "inspect_values",
        "rethink_logic",
    ] = Field(description="What should happen next.")
    diagnosis: str = Field(description="One sentence explaining the verdict.")
    suspect_table: Optional[str] = Field(
        default=None, description="Table to inspect, when next_action needs one."
    )
    suspect_column: Optional[str] = Field(
        default=None, description="Column to inspect, when next_action needs one."
    )


# ---------------------------------------------------------------------------
# 3. State
# ---------------------------------------------------------------------------

def _append(existing: list, new: list) -> list:
    return (existing or []) + (new or [])


class AgentState(TypedDict, total=False):
    question: str
    db_name: str
    evidence: str            # BIRD ships a domain hint with each question
    schema: str

    sql: str                 # current candidate
    result: dict[str, Any]   # last execution result
    assessment: dict[str, Any]

    findings: Annotated[list[str], _append]   # what investigate learned
    history: Annotated[list[dict], _append]   # every attempt, for the UI

    attempts: int
    finished: bool
    give_up_reason: str


# ---------------------------------------------------------------------------
# 4. Prompts
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = """You write SQLite queries.

Rules:
- Output exactly one SELECT statement. Never INSERT, UPDATE, DELETE or DROP.
- Use only tables and columns that appear in the schema given to you.
- Quote identifiers containing spaces or punctuation with double quotes.
  Column names in these databases often contain spaces and parentheses.
- Any division meant to produce a fraction must cast first:
  CAST(a AS REAL) / b. SQLite truncates integer division, so 5 / 10 is 0,
  not 0.5. This silently produces wrong answers for rates and percentages.
- Return exactly the columns the question asks for, in the order asked, and
  nothing else. A question asking "how many" wants one count, not a list.
- When a question mentions an attribute stored in a different table, join to
  that table rather than guessing a similarly-named column.
- Do not wrap the SQL in markdown fences."""


def generate_prompt(state: AgentState) -> list:
    parts = [
        f"Database: {state['db_name']}",
        f"\nSchema:\n{state['schema']}",
        f"\nQuestion: {state['question']}",
    ]
    if state.get("evidence"):
        parts.append(f"\nDomain hint: {state['evidence']}")

    if state.get("findings"):
        parts.append(
            "\nPrevious attempts failed. What was learned:\n"
            + "\n".join(f"- {f}" for f in state["findings"])
            + "\n\nWrite a corrected query that accounts for all of the above."
        )

    return [SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content="\n".join(parts))]


ASSESS_SYSTEM = """You judge whether a SQL attempt succeeded, and if not, what
kind of repair it needs. Choose exactly one next_action:

- done            : the query ran and the rows plausibly answer the question.
- fix_syntax      : malformed SQL. The intent is fine; the statement is not.
- fix_schema      : a referenced table or column does not exist.
- inspect_values  : the filter is comparing against a literal you cannot
                    verify, and the result suggests nothing matched. Set
                    suspect_table and suspect_column to the filtered column.
- rethink_logic   : the query ran and returned rows, but they do not answer the
                    question - wrong aggregation, wrong join, wrong grouping.

Two traps to watch for:

1. An aggregate hides an empty match. COUNT(*) over a filter that matches
   nothing returns ONE row containing 0, not zero rows - and SUM/AVG return one
   row containing NULL. So a result of 0 or NULL from an aggregate is the same
   signal as an empty result set: suspect the filter literal, not the syntax.

2. An empty result is not automatically wrong. If the question genuinely has no
   matching rows, answer done. But when the filter compares against a string
   literal that was guessed rather than read from the data, prefer
   inspect_values - checking costs one cheap lookup, and a wrong literal is the
   single most common cause of a silently wrong answer."""


def assess_prompt(state: AgentState) -> list:
    result = state.get("result", {})
    if result.get("success"):
        rows = result.get("rows", [])
        outcome = (
            f"Query ran successfully.\n"
            f"Columns: {result.get('columns')}\n"
            f"Rows returned: {result.get('row_count')}\n"
            f"Sample: {json.dumps(rows[:5], default=str)}"
        )
        # A single row of 0/NULL from an aggregate is an empty match wearing a
        # disguise. Say so explicitly rather than hoping the model notices.
        if len(rows) == 1 and all(
            c is None or c == 0 or c == "0" for c in rows[0]
        ):
            outcome += (
                "\nNOTE: this is a single row of zero/NULL values, which is what "
                "an aggregate returns when the filter matched nothing. Treat it "
                "like an empty result."
            )
    else:
        outcome = (
            f"Query failed.\n"
            f"Error class: {result.get('error_class')}\n"
            f"Error: {result.get('error')}"
        )

    return [
        SystemMessage(content=ASSESS_SYSTEM),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"SQL attempted:\n{state.get('sql')}\n\n"
            f"{outcome}"
        )),
    ]


# ---------------------------------------------------------------------------
# 5. Agent
# ---------------------------------------------------------------------------

class Text2SQLAgent:
    """Owns the MCP connection, the LLM bindings and the compiled graph."""

    def __init__(self, server_path: str = "db_server.py"):
        self.client = MultiServerMCPClient({
            "db": {
                "transport": "stdio",
                "command": "python",
                "args": [server_path],
            }
        })
        self.tools: dict[str, Any] = {}
        self.llm = build_llm()
        self.generator = self.llm.with_structured_output(GeneratedSQL)
        self.assessor = self.llm.with_structured_output(Assessment)
        self.graph = self._build_graph()
        self._schema_cache: dict[str, str] = {}

    # -- MCP plumbing ------------------------------------------------------

    async def _load_tools(self) -> None:
        if not self.tools:
            self.tools = {t.name: t for t in await self.client.get_tools()}

    async def _call(self, tool_name: str, **kwargs) -> dict:
        await self._load_tools()
        raw = await self.tools[tool_name].ainvoke(kwargs)
        return _parse_tool_result(raw)

    async def _get_schema(self, db_name: str) -> str:
        if db_name not in self._schema_cache:
            schema = await self._call("get_schema", db_name=db_name)
            self._schema_cache[db_name] = _format_schema(schema)
        return self._schema_cache[db_name]

    # -- nodes -------------------------------------------------------------

    async def load_schema_node(self, state: AgentState) -> dict:
        return {"schema": await self._get_schema(state["db_name"]), "attempts": 0}

    async def generate_node(self, state: AgentState) -> dict:
        prompt = generate_prompt(state)
        try:
            out: GeneratedSQL = await with_retry(
                lambda: self.generator.ainvoke(prompt)
            )
            sql = out.sql
        except Exception as exc:
            if not _is_structured_output_failure(exc):
                raise
            # The provider choked on wrapping the SQL in JSON. Ask for plain
            # text instead - we only need the statement, not a schema around it.
            raw = await with_retry(lambda: self.llm.ainvoke(
                prompt + [HumanMessage(content=(
                    "Reply with the SQL statement only. No JSON, no function "
                    "call, no markdown fences, no explanation."
                ))]
            ))
            sql = _extract_sql(getattr(raw, "content", str(raw)))

        sql = _clean_sql(sql)
        return {"sql": sql, "attempts": state.get("attempts", 0) + 1}

    async def execute_node(self, state: AgentState) -> dict:
        result = await self._call(
            "execute_query", db_name=state["db_name"], sql=state["sql"]
        )
        return {
            "result": result,
            "history": [{
                "attempt": state.get("attempts", 0),
                "sql": state["sql"],
                "success": result.get("success"),
                "row_count": result.get("row_count"),
                "error": result.get("error"),
            }],
        }

    async def assess_node(self, state: AgentState) -> dict:
        result = state.get("result", {})

        # Unambiguous failures do not need the model's opinion. Skipping the
        # call here saves roughly a third of the tokens on a failing run.
        error_class = result.get("error_class")
        shortcut = {
            "syntax_error": "fix_syntax",
            "unknown_table": "fix_schema",
            "unknown_column": "fix_schema",
            "ambiguous_column": "fix_syntax",
        }.get(error_class)

        if shortcut:
            return {"assessment": {
                "next_action": shortcut,
                "diagnosis": f"{error_class}: {result.get('error')}",
                "suspect_table": None,
                "suspect_column": None,
            }}

        prompt = assess_prompt(state)
        try:
            verdict: Assessment = await with_retry(
                lambda: self.assessor.ainvoke(prompt)
            )
            return {"assessment": verdict.model_dump()}
        except Exception as exc:
            if not _is_structured_output_failure(exc):
                raise
            raw = await with_retry(lambda: self.llm.ainvoke(
                prompt + [HumanMessage(content=(
                    "Reply with exactly one word, the next_action: done, "
                    "fix_syntax, fix_schema, inspect_values or rethink_logic."
                ))]
            ))
            return {"assessment": _parse_assessment_text(
                getattr(raw, "content", str(raw))
            )}

    async def investigate_node(self, state: AgentState) -> dict:
        """Gather the evidence the diagnosed failure class calls for."""
        assessment = state.get("assessment", {})
        action = assessment.get("next_action")
        diagnosis = assessment.get("diagnosis", "")
        findings = [f"Attempt {state.get('attempts')} failed: {diagnosis}"]

        if action == "inspect_values":
            table = assessment.get("suspect_table")
            column = assessment.get("suspect_column")
            if table and column:
                sample = await self._call(
                    "sample_column_values",
                    db_name=state["db_name"], table=table, column=column,
                )
                if "sample_values" in sample:
                    findings.append(
                        f'Actual values in "{table}"."{column}": '
                        f'{sample["sample_values"]} '
                        f'({sample.get("distinct_count")} distinct). '
                        f"Match your filter against these exactly."
                    )
                else:
                    findings.append(f"Could not sample {table}.{column}: "
                                    f"{sample.get('error')}")

        elif action == "fix_schema":
            # Telling the model "that column doesn't exist" is useless on its
            # own. Look up what the columns actually are and hand them over.
            error = state.get("result", {}).get("error", "")
            table = assessment.get("suspect_table") or _table_from_sql(
                state.get("sql", "")
            )
            resolved = False

            if table:
                info = await self._call(
                    "check_columns_exist",
                    db_name=state["db_name"], table=table, columns=[],
                )
                if info.get("all_columns"):
                    findings.append(
                        f'Real columns in "{table}": {info["all_columns"]}. '
                        f"Use these names exactly, including spaces and "
                        f"capitalisation, quoted with double quotes."
                    )
                    resolved = True
                elif info.get("available_tables"):
                    findings.append(
                        f'Table "{table}" does not exist. Real tables: '
                        f'{info["available_tables"]}.'
                    )
                    resolved = True

            if not resolved:
                findings.append(
                    f"A referenced table or column does not exist ({error}). "
                    f"Re-read the schema above and use only names that appear "
                    f"in it verbatim. If the attribute you need is not there, "
                    f"it probably lives in another table - join to it."
                )

        elif action == "fix_syntax":
            findings.append(f"Fix the syntax but keep the same query logic. "
                            f"Failed SQL was:\n{state.get('sql')}")

        elif action == "rethink_logic":
            findings.append(
                f"The query ran but the result did not answer the question. "
                f"Reconsider the joins, grouping and aggregation. "
                f"Previous SQL:\n{state.get('sql')}"
            )

        return {"findings": findings}

    # -- routing -----------------------------------------------------------

    def route_after_assess(self, state: AgentState) -> str:
        """
        The conditional edge. This is where the error class turns into a
        decision - the piece a plain retry loop does not have.
        """
        action = state.get("assessment", {}).get("next_action")

        if action == "done":
            return "finish"
        if state.get("attempts", 0) >= MAX_ATTEMPTS:
            return "exhausted"
        return "investigate"

    async def finish_node(self, state: AgentState) -> dict:
        return {"finished": True}

    async def exhausted_node(self, state: AgentState) -> dict:
        return {
            "finished": False,
            "give_up_reason": (
                f"Stopped after {state.get('attempts')} attempts. "
                f"Last diagnosis: {state.get('assessment', {}).get('diagnosis')}"
            ),
        }

    # -- graph -------------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("load_schema", self.load_schema_node)
        g.add_node("generate", self.generate_node)
        g.add_node("execute", self.execute_node)
        g.add_node("assess", self.assess_node)
        g.add_node("investigate", self.investigate_node)
        g.add_node("finish", self.finish_node)
        g.add_node("exhausted", self.exhausted_node)

        g.add_edge(START, "load_schema")
        g.add_edge("load_schema", "generate")
        g.add_edge("generate", "execute")
        g.add_edge("execute", "assess")

        g.add_conditional_edges(
            "assess",
            self.route_after_assess,
            {
                "finish": "finish",
                "exhausted": "exhausted",
                "investigate": "investigate",
            },
        )

        g.add_edge("investigate", "generate")   # the loop
        g.add_edge("finish", END)
        g.add_edge("exhausted", END)

        return g.compile()

    # -- public API --------------------------------------------------------

    async def ask(self, question: str, db_name: str, evidence: str = "") -> dict:
        """Answer one question. Returns the final state."""
        return await self.graph.ainvoke({
            "question": question,
            "db_name": db_name,
            "evidence": evidence,
            "findings": [],
            "history": [],
        })

    async def ask_single_shot(self, question: str, db_name: str,
                              evidence: str = "") -> dict:
        """
        Baseline: generate once, execute once, no repair. Used by evaluate.py to
        measure what the loop is actually worth.
        """
        schema = await self._get_schema(db_name)
        state: AgentState = {
            "question": question, "db_name": db_name,
            "evidence": evidence, "schema": schema, "findings": [],
        }
        gen = await self.generate_node(state)
        state.update(gen)
        exec_result = await self._call(
            "execute_query", db_name=db_name, sql=state["sql"]
        )
        return {"sql": state["sql"], "result": exec_result,
                "history": [{"attempt": 1, "sql": state["sql"],
                             "success": exec_result.get("success")}]}


# ---------------------------------------------------------------------------
# 6. Helpers
# ---------------------------------------------------------------------------

def _clean_sql(sql: str) -> str:
    """Strip markdown fences and stray whitespace from a generated statement."""
    sql = (sql or "").strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1] if "\n" in sql else sql[3:]
    sql = sql.removesuffix("```").strip()
    return sql.removeprefix("sql").strip() if sql.lower().startswith("sql\n") \
        else sql


def _extract_sql(text: str) -> str:
    """
    Pull a SELECT statement out of free-form model output.

    Used on the fallback path, where the model may add a sentence of
    explanation despite being asked not to.
    """
    import re

    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text or "",
                       re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    select = re.search(r"\b(SELECT|WITH)\b.+", text or "",
                       re.DOTALL | re.IGNORECASE)
    return select.group(0).strip() if select else (text or "").strip()


def _parse_assessment_text(text: str) -> dict:
    """Recover a verdict from a plain-text reply on the fallback path."""
    lowered = (text or "").lower()
    for action in ("inspect_values", "fix_schema", "fix_syntax",
                   "rethink_logic", "done"):
        if action in lowered:
            return {"next_action": action,
                    "diagnosis": (text or "").strip()[:200],
                    "suspect_table": None, "suspect_column": None}
    # Nothing recognisable: treat as done rather than looping pointlessly.
    return {"next_action": "done", "diagnosis": "unparsed verdict",
            "suspect_table": None, "suspect_column": None}


def _table_from_sql(sql: str) -> Optional[str]:
    """
    Pull the first table name out of a FROM or JOIN clause.

    Used when a query fails on an unknown column but the assessor did not say
    which table to inspect - a rough guess is better than no lookup at all.
    """
    import re

    match = re.search(r'(?:from|join)\s+[`"\[]?(\w+)[`"\]]?',
                      sql or "", re.IGNORECASE)
    return match.group(1) if match else None


def _parse_tool_result(raw: Any) -> dict:
    """
    Normalise whatever the MCP adapter hands back into a dict.

    Older adapter versions return the tool's string directly; newer ones return
    a list of content blocks like [{"type": "text", "text": "..."}]. Handling
    both keeps this working across versions.
    """
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, list):
        texts = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        raw = "".join(texts)

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    return {"raw": str(raw)}


def _format_schema(schema: dict) -> str:
    """Render the schema JSON as compact DDL-ish text - cheaper in tokens."""
    if "error" in schema:
        return f"ERROR: {schema['error']}"

    lines = []
    for table, info in schema.get("tables", {}).items():
        cols = ", ".join(
            f"{c['name']} {c['type']}" + (" PK" if c["primary_key"] else "")
            for c in info["columns"]
        )
        lines.append(f"TABLE {table} ({info['row_count']} rows)")
        lines.append(f"  {cols}")
        for fk in info["foreign_keys"]:
            lines.append(
                f"  FK {fk['column']} -> "
                f"{fk['references_table']}.{fk['references_column']}"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

async def _main() -> None:
    import sys

    if len(sys.argv) < 3:
        print("usage: python agent.py <db_name> <question>")
        print('example: python agent.py california_schools '
              '"How many schools are in Alameda County?"')
        return

    db_name, question = sys.argv[1], " ".join(sys.argv[2:])
    agent = Text2SQLAgent()
    final = await agent.ask(question, db_name)

    print(f"\nQuestion: {question}\n")
    for step in final.get("history", []):
        status = "ok" if step["success"] else f"failed - {step.get('error')}"
        print(f"  attempt {step['attempt']}: {status}")
        print(f"    {step['sql']}\n")

    if final.get("finished"):
        result = final.get("result", {})
        print(f"Answer ({result.get('row_count')} rows):")
        print(f"  {result.get('columns')}")
        for row in result.get("rows", [])[:10]:
            print(f"  {row}")
    else:
        print(final.get("give_up_reason"))


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())