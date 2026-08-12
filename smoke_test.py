"""
Verify the install without a BIRD download or an API key.

Builds two tiny databases plus a question file, then runs the agent with a
scripted stand-in for the model. Confirms that MCP transport, the graph, the
repair loop and the scoring harness all work before you spend real API calls.

    python smoke_test.py

Delete data/*.sqlite afterwards, or just let setup_data.py overwrite them.
"""

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("GOOGLE_API_KEY", "not-used-by-this-test")

import agent as A  # noqa: E402

DATA = Path("data")


def build_fixtures() -> None:
    DATA.mkdir(exist_ok=True)

    schools = DATA / "california_schools.sqlite"
    schools.unlink(missing_ok=True)
    conn = sqlite3.connect(schools)
    conn.executescript("""
        CREATE TABLE schools (CDSCode TEXT PRIMARY KEY, School TEXT,
                              County TEXT, City TEXT, Charter INTEGER);
        CREATE TABLE satscores (cds TEXT, NumTstTakr INTEGER,
                                AvgScrMath INTEGER, AvgScrRead INTEGER,
            FOREIGN KEY (cds) REFERENCES schools(CDSCode));
        INSERT INTO schools VALUES
         ('001','Berkeley High','Alameda County','Berkeley',0),
         ('002','Oakland Tech','Alameda County','Oakland',0),
         ('003','Lowell High','San Francisco County','San Francisco',0),
         ('004','KIPP Academy','Alameda County','Oakland',1);
        INSERT INTO satscores VALUES
         ('001',420,610,595),('002',380,548,560),
         ('003',510,702,688),('004',95,505,530);
    """)
    conn.commit()
    conn.close()

    questions = [
        {"db_id": "california_schools",
         "question": "How many schools are in Alameda County?",
         "evidence": "",
         "SQL": "SELECT COUNT(*) FROM schools WHERE County = 'Alameda County'"},
        {"db_id": "california_schools",
         "question": "List charter schools.",
         "evidence": "Charter = 1 means charter school",
         "SQL": "SELECT School FROM schools WHERE Charter = 1"},
    ]
    (DATA / "questions.json").write_text(json.dumps(questions, indent=1))
    print(f"  fixtures: 1 database, {len(questions)} questions")


class ScriptedGenerator:
    """
    Stands in for the LLM. Gets the county filter wrong the first time - the
    exact failure the repair loop exists to catch - then fixes it once the
    investigate node has supplied the real column values.
    """

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        text = messages[-1].content
        retrying = "Previous attempts failed" in text

        if retrying and "Alameda County" in text:
            sql = "SELECT COUNT(*) FROM schools WHERE County = 'Alameda County'"
        else:
            sql = "SELECT COUNT(*) FROM schools WHERE County = 'Alameda'"
        return A.GeneratedSQL(sql=sql, reasoning="scripted")


class ScriptedAssessor:
    async def ainvoke(self, messages):
        text = messages[-1].content
        empty = ("Rows returned: 0" in text
                 or "single row of zero/NULL" in text)
        if empty:
            return A.Assessment(
                next_action="inspect_values",
                diagnosis="nothing matched - filter literal is suspect",
                suspect_table="schools", suspect_column="County",
            )
        return A.Assessment(next_action="done", diagnosis="plausible result")


async def main() -> int:
    print("Building fixtures...")
    build_fixtures()

    print("\nStarting agent (launches db_server.py over MCP stdio)...")
    agent = A.Text2SQLAgent()
    agent.generator = ScriptedGenerator()
    agent.assessor = ScriptedAssessor()

    print("\nSchema read through MCP:")
    schema = await agent._get_schema("california_schools")
    for line in schema.splitlines()[:6]:
        print(f"  {line}")

    print("\nRunning the repair loop...")
    final = await agent.ask("How many schools are in Alameda County?",
                            "california_schools")

    for step in final.get("history", []):
        status = "ok" if step["success"] else f"error: {step.get('error')}"
        print(f"  attempt {step['attempt']}: rows={step.get('row_count')} "
              f"({status})")
        print(f"    {step['sql']}")

    print("\n  What it learned in between:")
    for finding in final.get("findings", []):
        print(f"    - {finding}")

    rows = final.get("result", {}).get("rows", [])
    checks = {
        "finished": final.get("finished") is True,
        "took more than one attempt": final.get("attempts", 0) > 1,
        "final answer is 3": bool(rows) and rows[0][0] == 3,
        "sampled the real column values": any(
            "Alameda County" in f for f in final.get("findings", [])
        ),
    }

    print()
    for name, passed in checks.items():
        print(f"  [{'pass' if passed else 'FAIL'}] {name}")

    if all(checks.values()):
        print("\nEverything works. Add your API key to .env and run:")
        print("  python evaluate.py --limit 5")
        return 0

    print("\nSomething is wrong - see the failures above.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
