"""
Prepare the BIRD Mini-Dev data.

BIRD does not offer a stable direct-download URL, so download the zip manually
(link in data/README.md), then point this script at the extracted folder:

    python setup_data.py --source ~/Downloads/minidev

It copies out the two databases we use, filters the question file to match, and
verifies everything loads.
"""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

DATA_DIR = Path("data")

# Two databases is enough to show the agent generalises across schemas without
# turning setup into a project of its own.
KEEP = ["california_schools", "financial"]


def find_databases(source: Path) -> dict[str, Path]:
    found = {}
    for path in source.rglob("*.sqlite"):
        if path.stem in KEEP:
            found[path.stem] = path
    return found


def find_questions(source: Path) -> Path | None:
    candidates = [
        p for p in source.rglob("*.json")
        if "question" in p.name.lower() or "mini_dev" in p.name.lower()
    ]
    for path in candidates:
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list) and data and "db_id" in data[0]:
                return path
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="folder containing the extracted Mini-Dev download")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        raise SystemExit(f"{source} does not exist")

    DATA_DIR.mkdir(exist_ok=True)

    # databases
    databases = find_databases(source)
    missing = [name for name in KEEP if name not in databases]
    if missing:
        raise SystemExit(
            f"could not find {missing} under {source}. "
            f"Did you extract dev_databases.zip inside the download?"
        )

    for name, path in databases.items():
        target = DATA_DIR / f"{name}.sqlite"
        shutil.copy(path, target)
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        conn.close()
        size_mb = target.stat().st_size / 1e6
        print(f"  {name}: {tables} tables, {size_mb:.1f} MB")

    # questions
    questions_path = find_questions(source)
    if not questions_path:
        raise SystemExit(f"no question JSON found under {source}")

    all_questions = json.loads(questions_path.read_text())
    filtered = [q for q in all_questions if q.get("db_id") in KEEP]

    out = DATA_DIR / "questions.json"
    out.write_text(json.dumps(filtered, indent=1))

    by_db = {}
    for q in filtered:
        by_db[q["db_id"]] = by_db.get(q["db_id"], 0) + 1

    print(f"\n  {len(filtered)} questions kept "
          f"(from {len(all_questions)} total)")
    for db, count in sorted(by_db.items()):
        print(f"    {db}: {count}")
    print(f"\nWrote {out}. Ready to run:  python evaluate.py --limit 5")


if __name__ == "__main__":
    main()
