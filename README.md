# Text-to-SQL Agent with Error-Class-Conditioned Repair

A natural-language interface to SQL databases that **executes what it writes**,
inspects what came back, and repairs itself based on the *kind* of failure.

A chat model hands you SQL and stops. It never finds out that
`WHERE County = 'Alameda'` returns zero rows because the column actually holds
`'Alameda County'`. This agent runs the query, sees the empty result, samples
the real column values, and rewrites the filter.

Evaluated on the BIRD benchmark.

---

## The idea

Most agents retry blindly on failure: same prompt, same context, hoping for a
different sample. But the failure itself is the most informative signal
available, and it's thrown away.

| What came back | What it means | Repair |
|---|---|---|
| Syntax error | Statement is malformed, logic is fine | Fix the SQL, keep the intent |
| Unknown column | The schema was misread | Re-check the schema, regenerate |
| Empty result | Filter compared against a guessed literal | **Go look at the actual values**, then rewrite |
| Rows, but wrong | Join or aggregation is wrong | Reconsider the approach |

Those need four different responses. Treating them identically is why naive
retry underperforms.

## Graph

```
                    ┌──────────────┐
                    │ load_schema  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
              ┌────►│   generate   │
              │     └──────┬───────┘
              │            ▼
              │     ┌──────────────┐
              │     │   execute    │  ── via MCP, read-only
              │     └──────┬───────┘
              │            ▼
              │     ┌──────────────┐
              │     │    assess    │  ── structured verdict
              │     └──────┬───────┘
              │            │
              │     ┌──────┴─────────────────┐
              │     │  conditional routing   │
              │     └──┬──────────┬──────┬───┘
              │        │          │      │
              │    done│  exhausted      │ repair needed
              │        ▼          ▼      ▼
              │       END        END  ┌──────────────┐
              └───────────────────────┤ investigate  │
                                      └──────────────┘
```

`investigate` gathers whatever the diagnosis calls for — sampling column values,
re-reading the schema, or restating the failed logic — and hands that back to
`generate` as evidence.

## Architecture

The agent has **no direct database access**. Everything goes through an MCP
server:

| Tool | Purpose |
|---|---|
| `get_schema` | Tables, columns, types, foreign keys, row counts |
| `execute_query` | Read-only execution with a timeout and a classified error |
| `sample_column_values` | What's *actually* in a column — the repair loop's key input |
| `check_columns_exist` | Cheap validation before executing |
| `list_databases` | Discovery |

Because the boundary is a protocol rather than a function call, the agent
contains zero database-specific logic — point the server at a different SQLite
file and it works unchanged.

Execution is read-only (`mode=ro`), timed out, and row-capped. Generated SQL is
untrusted by construction.

## Results

See [`results/summary.md`](results/summary.md).

```
Evaluated on N BIRD Mini-Dev questions across 2 databases.

                        Single-shot    With repair loop
Execution accuracy         --.-%            --.-%
Ran without error          --.-%            --.-%
Average attempts            1.00             -.--
```

Run `python evaluate.py --mode both` to populate this.

## Setup

```bash
git clone <this-repo> && cd text2sql-agent
pip install -r requirements.txt

cp .env.example .env        # add your GOOGLE_API_KEY
```

Get a free Gemini key at https://aistudio.google.com/apikey — no card required.

Then follow [`data/README.md`](data/README.md) to download the BIRD Mini-Dev
databases and run:

```bash
python setup_data.py --source ~/Downloads/minidev
```

## Usage

**One question from the CLI:**

```bash
python agent.py california_schools "How many schools are in Alameda County?"
```

**The UI:**

```bash
streamlit run app.py
```

**Evaluate:**

```bash
python smoke_test.py              # verify install, no API key needed
python evaluate.py --limit 5      # real run, uses API
python evaluate.py --mode both    # full run, both modes
```

**Debug the MCP server on its own:**

```bash
python db_server.py
```

## Files

```
db_server.py     MCP server — the five database tools
agent.py         State, nodes, routing, graph
evaluate.py      Execution-accuracy harness
app.py           Streamlit UI
setup_data.py    One-time data preparation
smoke_test.py    Verifies the install with no API key and no download
```

## Notes

- **Cost: zero.** Gemini's free tier covers a full evaluation run comfortably.
- **No GPU.** Everything runs on a laptop.
- Scoring is execution accuracy — result sets are compared, not SQL strings,
  since many different queries are correct.
- Schemas are passed in full rather than retrieved. At Mini-Dev's scale they fit
  in context, and adding retrieval would be complexity without benefit.

## Built with

LangGraph · MCP (FastMCP) · Gemini / Groq · SQLite · Streamlit
