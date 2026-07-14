"""
Merge two MLflow SQLite databases by appending runs from an old DB into a new one.

Usage:
    uv run python scripts/merge_mlflow.py --old mlflow_old.db --new mlflow.db

The NEW db is modified in place. The OLD db is read-only.
Runs that already exist in NEW (by run_uuid) are skipped.
Experiment IDs are remapped by experiment NAME.
"""
import argparse
import sqlite3
from pathlib import Path


def _exp_name_to_id(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        name: eid
        for eid, name in conn.execute(
            "SELECT experiment_id, name FROM experiments WHERE lifecycle_stage = 'active'"
        ).fetchall()
    }


def merge(old_path: Path, new_path: Path) -> None:
    if not old_path.exists():
        raise FileNotFoundError(old_path)
    if not new_path.exists():
        raise FileNotFoundError(new_path)

    old = sqlite3.connect(old_path)
    new = sqlite3.connect(new_path)

    old_exp_map = _exp_name_to_id(old)   # name → old experiment_id
    new_exp_map = _exp_name_to_id(new)   # name → new experiment_id

    # Build remapping: old experiment_id → new experiment_id
    exp_remap: dict[int, int] = {}
    for name, old_id in old_exp_map.items():
        if name in new_exp_map:
            exp_remap[old_id] = new_exp_map[name]
        else:
            # Experiment doesn't exist in new db — create it
            row = old.execute(
                "SELECT name, artifact_location, lifecycle_stage, creation_time, last_update_time "
                "FROM experiments WHERE experiment_id = ?", (old_id,)
            ).fetchone()
            cur = new.execute(
                "INSERT INTO experiments (name, artifact_location, lifecycle_stage, creation_time, last_update_time) "
                "VALUES (?, ?, ?, ?, ?)", row
            )
            exp_remap[old_id] = cur.lastrowid
            print(f"  Created experiment '{name}' (new id={cur.lastrowid})")

    # Existing run UUIDs in new db
    existing_runs = {
        r[0] for r in new.execute("SELECT run_uuid FROM runs").fetchall()
    }

    # Copy runs not present in new db
    runs_copied = 0
    old_runs = old.execute(
        "SELECT run_uuid, name, source_type, source_name, entry_point_name, "
        "user_id, status, start_time, end_time, source_version, lifecycle_stage, "
        "artifact_uri, experiment_id, deleted_time FROM runs"
    ).fetchall()

    for row in old_runs:
        run_uuid = row[0]
        if run_uuid in existing_runs:
            continue

        old_exp_id = row[12]
        new_exp_id = exp_remap.get(old_exp_id)
        if new_exp_id is None:
            print(f"  SKIP run {run_uuid}: experiment_id {old_exp_id} not mapped")
            continue

        new_row = row[:12] + (new_exp_id,) + row[13:]
        new.execute(
            "INSERT INTO runs (run_uuid, name, source_type, source_name, entry_point_name, "
            "user_id, status, start_time, end_time, source_version, lifecycle_stage, "
            "artifact_uri, experiment_id, deleted_time) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", new_row
        )
        runs_copied += 1

        # Copy metrics, params, tags, latest_metrics for this run
        for table, cols in [
            ("metrics",        "key, value, timestamp, run_uuid, step, is_nan"),
            ("params",         "key, value, run_uuid"),
            ("tags",           "key, value, run_uuid"),
            ("latest_metrics", "key, value, timestamp, run_uuid, step, is_nan"),
        ]:
            # Check table exists in old db
            exists = old.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            ).fetchone()
            if not exists:
                continue
            rows = old.execute(f"SELECT {cols} FROM {table} WHERE run_uuid = ?", (run_uuid,)).fetchall()
            if rows:
                placeholders = ",".join("?" * len(rows[0]))
                new.executemany(f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})", rows)

    new.commit()
    old.close()
    new.close()
    print(f"\nDone. {runs_copied} runs copied from {old_path.name} into {new_path.name}.")
    if runs_copied == 0:
        print("(All runs from old db already existed in new db.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two MLflow SQLite databases")
    parser.add_argument("--old", type=Path, required=True, help="Source db (read-only)")
    parser.add_argument("--new", type=Path, required=True, help="Target db (modified in place)")
    args = parser.parse_args()
    print(f"Merging {args.old} → {args.new}")
    merge(args.old, args.new)
