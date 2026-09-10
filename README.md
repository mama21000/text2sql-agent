# Self-Correcting Text-to-SQL Agent

Ask a question in plain English, get an answer from your database.

Unlike a chat model that writes SQL and stops, this agent **executes what it
writes**, reads the failure, and repairs itself — routing each kind of failure
to a targeted fix rather than blindly retrying. A model handed a schema will
happily write `WHERE County = 'Alameda'` when the column actually stores
`'Alameda County'` — valid SQL, no error, zero rows, and nobody finds out. This
agent runs the query, sees the empty result, samples the real column values,
and rewrites the filter.

---

## The idea

Most agents retry blindly on failure: same prompt, hoping for a different
sample. But the failure itself tells you what to do.

| What came back | What it means | The repair |
|---|---|---|
| Syntax error | Statement malformed, intent was fine | fix the SQL, keep the logic |
| Unknown column | Schema was misread or hallucinated | look up the real column names |
| Empty result | Filter compared against a guessed literal | sample the column's actual values |
| Rows, but wrong | Wrong join or aggregation | reconsider the approach |

Four different failures, four different responses.

## How it works

```
   load_schema
        │
        ▼
   ┌─────────┐
┌─►│generate │  ask the model for SQL
│  └────┬────┘
│       ▼
│  ┌─────────┐
│  │ execute │  run it via MCP, read-only
│  └────┬────┘
│       ▼
│  ┌─────────┐
│  │ assess  │  structured verdict: done, or which repair
│  └────┬────┘
│       │
│  ┌────┴─────────────┐
│  │ conditional edge  │
│  └──┬────────┬───────┘
│    done   repair needed
│     ▼         ▼
│    END  ┌────────────┐
└─────────┤ investigate│  gather the evidence this failure needs
          └────────────┘
```

`investigate` does the work that makes repair possible — calling
`sample_column_values` when a filter matched nothing, or `check_columns_exist`
when a column was hallucinated. What it learns accumulates in graph state and
is fed back into the next generation attempt.

The model is stateless between calls. The graph is what remembers, and it
re-tells the model everything relevant on every attempt.

## Architecture

The agent has **no direct database access**. Everything goes through an MCP
server running as a separate process:

| Tool | Purpose |
|---|---|
| `get_schema` | Tables, columns, types, foreign keys, row counts |
| `execute_query` | Read-only execution, returns rows or a classified error |
| `sample_column_values` | What's actually stored in a column |
| `check_columns_exist` | Real column names when one was hallucinated |
| `list_databases` | Discovery |

Because the boundary is a protocol rather than a function call, `agent.py`
contains zero database-specific logic. Point the server at a different
database and the agent works unchanged — porting to PostgreSQL means rewriting
the schema-introspection queries in `db_server.py` alone.

## Safety

Generated SQL is untrusted by construction:

- **Read-only connections** (`mode=ro`) — enforced by SQLite itself, not by
  prompt instruction. `DELETE` and `DROP` are rejected at the driver level.
- **Query timeouts** — 15 seconds, then the query is interrupted.
- **Row caps** — 50 rows maximum per result.
- **Path sanitisation** — database names containing `/`, `\` or `..` are refused.
- **Process isolation** — execution happens in a different process from the
  agent's own logic.

## Results

Evaluated on 20 BIRD Mini-Dev questions across two databases
(`california_schools`, `financial`). Both modes run the same questions with the
same prompts, schema and MCP server — the only variable is the repair loop.

**Llama 3.1 8B** — a weak model that frequently produced invalid SQL:

| | Single-shot | With repair loop |
|---|---|---|
| Execution accuracy | 5% | **15%** |
| Queries that ran without error | 30% | **65%** |
| Average attempts | 1.00 | 2.70 |

**gpt-oss-120b** — a stronger model that produced valid SQL every time:

| | Single-shot | With repair loop |
|---|---|---|
| Execution accuracy | 20% | 20% |
| Queries that ran without error | 100% | 100% |
| Average attempts | 1.00 | 1.95 |

### What this shows

The loop repairs failures it can **see by executing** — malformed SQL,
hallucinated columns, filters that match nothing. On the weak model those were
70% of queries, and the loop recovered more than half of them, tripling accuracy.

On the strong model the loop gained nothing, and the reason is the interesting
part: every query executed successfully. Its 16 wrong answers were valid SQL
returning plausible-looking rows — wrong joins, wrong aggregations — which
execution feedback cannot detect. There is no error to read and no zero-row
signal to catch.

**Execution feedback repairs syntactic failures, not semantic ones.** Catching
the latter needs verification against the question rather than against the
database, which this design does not attempt.

### Scope

These numbers measure the effect of the repair loop, not competitive
performance. Published BIRD leaderboard systems reach 74–78% using frontier
models with schema retrieval, few-shot prompting and multi-candidate selection;
this uses free-tier models with the full schema in context and none of those
techniques. The absolute accuracy is a property of the model, not of the loop.

## Setup

```bash
git clone https://github.com/mama21000/text2sql-agent.git
cd text2sql-agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # add an API key
```

Free API keys, no card required:
[Google AI Studio](https://aistudio.google.com/apikey) ·
[Groq](https://console.groq.com)

Verify the install — no API key or dataset needed:

```bash
python smoke_test.py
```

Then follow [`data/README.md`](data/README.md) to download the databases and run:

```bash
python setup_data.py --source ~/Downloads/minidev_extracted
```

## Usage

```bash
# one question
python agent.py california_schools "How many schools are in Alameda County?"

# web interface
streamlit run app.py

# benchmark against BIRD's gold queries
python evaluate.py --mode both --limit 20
```

`--mode both` runs single-shot generation and the full repair loop over the
same questions, so the loop's contribution is measurable rather than assumed.
Scoring is execution accuracy — result sets are compared, not SQL text, since
many different queries are correct.

## Files

```
db_server.py     MCP server - the five database tools
agent.py         state, nodes, routing, the graph
evaluate.py      execution-accuracy harness
app.py           Streamlit interface
setup_data.py    one-time data preparation
smoke_test.py    verifies the install with no API key and no download
```

## Notes

- Schemas are passed in full rather than retrieved. At this scale they fit in
  context, and adding retrieval would be complexity without benefit.
- Structured output falls back to text parsing when a provider's tool-call
  layer rejects SQL containing string literals as invalid JSON — a real
  failure mode on some hosted models.
- Rate-limit retry with exponential backoff, since free-tier quotas throttle
  hard during full evaluation runs.

## Data

Evaluated against [BIRD Mini-Dev](https://bird-bench.github.io/)
(Li et al., 2023), licensed CC BY-SA 4.0.

## Built with

LangGraph · MCP (FastMCP) · Gemini / Groq · SQLite · Pydantic · Streamlit
