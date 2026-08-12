"""
MCP server exposing read-only database tools to the agent.

The agent has no direct database access. Every schema lookup, value sample and
query execution goes through this server, which means the agent code contains
zero database-specific logic - point this at a different SQLite file and the
agent works unchanged.

Run standalone for debugging:
    python db_server.py

Normally launched automatically by agent.py over stdio.
"""

import json
import os
import sqlite3
from pathlib import Path

# The SDK renamed FastMCP -> MCPServer in 2.0. Support both so the project
# doesn't break on whichever version happens to be installed.
try:
    from mcp.server.fastmcp import FastMCP as _Server      # mcp 1.x
except ImportError:                                        # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
QUERY_TIMEOUT_SECONDS = 15
MAX_ROWS_RETURNED = 50

mcp = _Server("db-server")


# ---------------------------------------------------------------------------
# connection helpers
# ---------------------------------------------------------------------------

def _db_path(db_name: str) -> Path:
    """Resolve a database name to a path, refusing anything outside DATA_DIR."""
    if "/" in db_name or "\\" in db_name or ".." in db_name:
        raise ValueError(f"invalid database name: {db_name}")
    path = DATA_DIR / f"{db_name}.sqlite"
    if not path.exists():
        available = sorted(p.stem for p in DATA_DIR.glob("*.sqlite"))
        raise FileNotFoundError(
            f"database '{db_name}' not found. available: {available}"
        )
    return path


def _connect(db_name: str) -> sqlite3.Connection:
    """Open a read-only connection. Generated SQL cannot modify anything."""
    path = _db_path(db_name)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    # Interrupt runaway queries: SQLite calls this every N VM instructions.
    conn.set_progress_handler(_make_watchdog(), 100_000)
    return conn


def _make_watchdog():
    """Return a progress callback that aborts the query after the timeout."""
    import time

    deadline = time.time() + QUERY_TIMEOUT_SECONDS

    def handler():
        return 1 if time.time() > deadline else 0

    return handler


def _classify_error(exc: Exception) -> str:
    """
    Map a database exception to one of a small set of classes.

    The agent routes its repair strategy on this value, so the categories matter
    more than the message text: a syntax error and a missing column need
    completely different fixes.
    """
    message = str(exc).lower()

    if "no such table" in message:
        return "unknown_table"
    if "no such column" in message:
        return "unknown_column"
    if "ambiguous column" in message:
        return "ambiguous_column"
    if "syntax error" in message or "incomplete input" in message:
        return "syntax_error"
    if "no such function" in message:
        return "unknown_function"
    if "interrupted" in message:
        return "timeout"
    if "misuse" in message or "readonly" in message:
        return "write_attempt"
    return "other_error"


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_databases() -> str:
    """List the database names available to query."""
    names = sorted(p.stem for p in DATA_DIR.glob("*.sqlite"))
    return json.dumps({"databases": names})


@mcp.tool()
def get_schema(db_name: str) -> str:
    """
    Return the full schema of a database: every table with its columns, types,
    primary keys and foreign keys. Call this before writing any query.
    """
    try:
        conn = _connect(db_name)
    except (ValueError, FileNotFoundError) as exc:
        return json.dumps({"error": str(exc)})

    try:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]

        schema = {}
        for table in tables:
            quoted = f'"{table}"'
            columns = [
                {
                    "name": row["name"],
                    "type": row["type"] or "UNKNOWN",
                    "primary_key": bool(row["pk"]),
                    "not_null": bool(row["notnull"]),
                }
                for row in conn.execute(f"PRAGMA table_info({quoted})")
            ]
            foreign_keys = [
                {
                    "column": row["from"],
                    "references_table": row["table"],
                    "references_column": row["to"],
                }
                for row in conn.execute(f"PRAGMA foreign_key_list({quoted})")
            ]
            row_count = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]

            schema[table] = {
                "columns": columns,
                "foreign_keys": foreign_keys,
                "row_count": row_count,
            }

        return json.dumps({"database": db_name, "tables": schema}, indent=1)
    finally:
        conn.close()


@mcp.tool()
def execute_query(db_name: str, sql: str) -> str:
    """
    Execute a read-only SQL query and return the resulting rows.

    On failure returns an 'error_class' field describing what kind of failure it
    was, so the caller can choose an appropriate repair rather than blindly
    retrying. An empty result set is reported as success with row_count 0 -
    that is usually a filter-value problem, not a syntax problem.
    """
    try:
        conn = _connect(db_name)
    except (ValueError, FileNotFoundError) as exc:
        return json.dumps({"success": False, "error_class": "bad_database",
                           "error": str(exc)})

    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(MAX_ROWS_RETURNED)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        truncated = len(cursor.fetchmany(1)) > 0

        return json.dumps({
            "success": True,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }, default=str)

    except Exception as exc:
        return json.dumps({
            "success": False,
            "error_class": _classify_error(exc),
            "error": str(exc),
        })
    finally:
        conn.close()


@mcp.tool()
def sample_column_values(db_name: str, table: str, column: str,
                         limit: int = 15) -> str:
    """
    Return distinct values actually stored in a column.

    Use this when a query runs without error but returns nothing. The usual
    cause is a filter written against a guessed value - 'Alameda' when the
    column really holds 'Alameda County'. Column names alone cannot tell you
    this; you have to look.
    """
    try:
        conn = _connect(db_name)
    except (ValueError, FileNotFoundError) as exc:
        return json.dumps({"error": str(exc)})

    try:
        limit = max(1, min(limit, 50))
        rows = conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT ?',
            (limit,),
        ).fetchall()
        distinct_total = conn.execute(
            f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
        ).fetchone()[0]

        return json.dumps({
            "table": table,
            "column": column,
            "sample_values": [r[0] for r in rows],
            "distinct_count": distinct_total,
        }, default=str)

    except Exception as exc:
        return json.dumps({"error": str(exc),
                           "error_class": _classify_error(exc)})
    finally:
        conn.close()


@mcp.tool()
def check_columns_exist(db_name: str, table: str, columns: list[str]) -> str:
    """
    Check whether columns exist on a table before running a query.

    Cheaper than executing and failing, and when a column is missing this
    returns the real column names so the caller can pick the right one.
    """
    try:
        conn = _connect(db_name)
    except (ValueError, FileNotFoundError) as exc:
        return json.dumps({"error": str(exc)})

    try:
        actual = [row["name"] for row in conn.execute(f'PRAGMA table_info("{table}")')]
        if not actual:
            tables = [
                r["name"] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            return json.dumps({"error": f"no such table: {table}",
                               "available_tables": tables})

        lowered = {c.lower(): c for c in actual}
        results = {c: lowered.get(c.lower()) for c in columns}

        return json.dumps({
            "table": table,
            "resolved": {k: v for k, v in results.items() if v},
            "missing": [k for k, v in results.items() if not v],
            "all_columns": actual,
        })
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run(transport="stdio")
