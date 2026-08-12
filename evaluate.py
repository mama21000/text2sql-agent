"""
Evaluate the agent on BIRD Mini-Dev questions.

Metric is execution accuracy: run the predicted SQL and the gold SQL against the
same database and compare the result sets. String-matching the SQL would be
wrong - there are many correct ways to write the same query.

Runs both modes so the loop's contribution is measurable:

    python evaluate.py --mode single_shot   # baseline, no repair
    python evaluate.py --mode agent         # with the repair loop
    python evaluate.py --mode both          # both, then a comparison table

Results land in results/ as JSON plus a readable summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from agent import Text2SQLAgent

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RESULTS_DIR = Path("results")
QUESTIONS_FILE = DATA_DIR / "questions.json"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def run_gold(db_name: str, sql: str) -> list | None:
    """Execute the reference query. None means the gold SQL itself failed."""
    path = DATA_DIR / f"{db_name}.sqlite"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        rows = conn.execute(sql).fetchall()
        conn.close()
        return rows
    except Exception:
        return None


def results_match(predicted: dict, gold_rows: list | None) -> bool:
    """
    Compare result sets, ignoring row order and column order.

    BIRD questions ask for values, not orderings, so two queries returning the
    same multiset of rows are both correct even if ORDER BY differs.
    """
    if gold_rows is None or not predicted.get("success"):
        return False

    pred_rows = predicted.get("rows", [])

    if predicted.get("truncated"):
        # We capped rows at 50 in the server; a truncated result can't be
        # compared safely, so count it wrong rather than guess.
        return False

    def normalise(rows):
        out = []
        for row in rows:
            cells = tuple(sorted(
                "" if c is None else str(c).strip().lower() for c in row
            ))
            out.append(cells)
        return sorted(out)

    return normalise(pred_rows) == normalise(gold_rows)


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

async def evaluate(mode: str, questions: list[dict], limit: int | None) -> dict:
    agent = Text2SQLAgent()
    if limit:
        questions = questions[:limit]

    records = []
    started = time.time()

    for i, item in enumerate(questions, 1):
        question = item["question"]
        db_name = item["db_id"]
        gold_sql = item.get("SQL") or item.get("gold_sql", "")
        evidence = item.get("evidence", "")

        try:
            if mode == "single_shot":
                out = await agent.ask_single_shot(question, db_name, evidence)
                predicted, sql = out["result"], out["sql"]
                attempts = 1
            else:
                out = await agent.ask(question, db_name, evidence)
                predicted, sql = out.get("result", {}), out.get("sql", "")
                attempts = out.get("attempts", 1)

            gold_rows = run_gold(db_name, gold_sql)
            correct = results_match(predicted, gold_rows)

            records.append({
                "index": i,
                "db_id": db_name,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": sql,
                "correct": correct,
                "attempts": attempts,
                "executed": bool(predicted.get("success")),
                "error_class": predicted.get("error_class"),
                "first_attempt_failed": (
                    bool(out.get("history")) and not out["history"][0].get("success")
                ),
            })

        except Exception as exc:                       # keep the run alive
            records.append({
                "index": i, "db_id": db_name, "question": question,
                "correct": False, "attempts": 0, "executed": False,
                "error_class": "agent_crash", "error": str(exc),
            })
        await asyncio.sleep(15)
        if i % 10 == 0 or i == len(questions):
            hits = sum(r["correct"] for r in records)
            print(f"  [{mode}] {i}/{len(questions)}  "
                  f"accuracy {hits / i:.1%}", flush=True)

    elapsed = time.time() - started
    correct = sum(r["correct"] for r in records)

    summary = {
        "mode": mode,
        "total": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "execution_rate": sum(r["executed"] for r in records) / len(records),
        "avg_attempts": sum(r["attempts"] for r in records) / len(records),
        "recovered": sum(
            1 for r in records if r["correct"] and r["attempts"] > 1
        ),
        "error_classes": dict(Counter(
            r["error_class"] for r in records if r.get("error_class")
        )),
        "elapsed_seconds": round(elapsed, 1),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{mode}.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    return summary


def write_summary(summaries: list[dict]) -> None:
    lines = ["# Results", ""]

    if len(summaries) == 2:
        base, full = summaries[0], summaries[1]
        delta = (full["accuracy"] - base["accuracy"]) * 100
        lines += [
            f"Evaluated on {full['total']} BIRD Mini-Dev questions.",
            "",
            "| | Single-shot | With repair loop |",
            "|---|---|---|",
            f"| Execution accuracy | {base['accuracy']:.1%} | {full['accuracy']:.1%} |",
            f"| Ran without error | {base['execution_rate']:.1%} | {full['execution_rate']:.1%} |",
            f"| Avg attempts | {base['avg_attempts']:.2f} | {full['avg_attempts']:.2f} |",
            "",
            f"The repair loop added **{delta:+.1f} points** of execution accuracy. "
            f"**{full['recovered']} of {full['correct']} correct answers** needed "
            f"more than one attempt — those are queries single-shot generation "
            f"got wrong.",
            "",
        ]

    for s in summaries:
        lines += [
            f"## {s['mode']}",
            "",
            f"- Accuracy: {s['correct']}/{s['total']} ({s['accuracy']:.1%})",
            f"- Ran without error: {s['execution_rate']:.1%}",
            f"- Average attempts: {s['avg_attempts']:.2f}",
            f"- Runtime: {s['elapsed_seconds']}s",
            "",
        ]
        if s["error_classes"]:
            lines.append("Failure classes:")
            lines += [f"- `{k}`: {v}" for k, v in
                      sorted(s["error_classes"].items(), key=lambda x: -x[1])]
            lines.append("")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single_shot", "agent", "both"],
                        default="both")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions")
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        print(f"{QUESTIONS_FILE} not found. See data/README.md for setup.")
        return

    questions = json.loads(QUESTIONS_FILE.read_text())
    print(f"Loaded {len(questions)} questions.\n")

    modes = ["single_shot", "agent"] if args.mode == "both" else [args.mode]
    summaries = [await evaluate(m, questions, args.limit) for m in modes]
    write_summary(summaries)


if __name__ == "__main__":
    asyncio.run(main())
