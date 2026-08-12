# Data setup

This folder is gitignored — the databases are downloaded, not committed.

## 1. Download BIRD Mini-Dev

Go to **https://bird-bench.github.io/** and download the **Mini-Dev** set
(500 question–SQL pairs across 11 databases, SQLite version).

The data is licensed CC BY-SA 4.0. No account, no API key.

If you only find the full `dev.zip`, that works too — it contains the same
databases, just more questions.

## 2. Extract

```bash
unzip minidev.zip -d ~/Downloads/minidev
cd ~/Downloads/minidev
unzip dev_databases.zip        # the databases are in a nested zip
```

## 3. Run the setup script

From the project root:

```bash
python setup_data.py --source ~/Downloads/minidev
```

This copies out the two databases the project uses and filters the question
file to match. You should end up with:

```
data/
├── california_schools.sqlite
├── financial.sqlite
├── questions.json
└── README.md
```

## Why only two databases?

Enough to show the agent generalises across different schemas without turning
setup into its own project. `california_schools` has messy real-world values
(good for triggering value-mismatch repairs); `financial` has more tables and
deeper joins.

To use more, edit `KEEP` in `setup_data.py`.
