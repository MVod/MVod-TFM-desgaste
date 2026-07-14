"""
train_final.py — Train each branch on all data and save a deployable model.

Usage:
    uv run python train_final.py --branch B
    uv run python train_final.py --branch all
"""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import yaml
from typing import Dict

from evaluation.loocv import build_tool_index
from inference_utils import model_path
from pipeline import load_branch, set_seed

BRANCHES = ["A", "B", "C"]


def train_branch_final(branch_name: str, config: Dict) -> Path:
    set_seed(config["seed"])
    data_dir = Path(config["data_dir"])
    tool_index = build_tool_index(data_dir)
    all_tool_ids = list(tool_index.keys())

    branch = load_branch(branch_name)

    print(f"[Branch {branch_name}] Training on {len(all_tool_ids)} tools...")
    branch.train(all_tool_ids, tool_index, config)

    out_path = model_path(branch_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    branch.save(out_path)
    print(f"[Branch {branch_name}] Saved → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final model for deployment")
    parser.add_argument("--branch", required=True, choices=BRANCHES + ["all"])
    args = parser.parse_args()

    branches = BRANCHES if args.branch == "all" else [args.branch]
    for b in branches:
        config_path = Path(f"configs/branch_{b.lower()}.yaml")
        config = yaml.safe_load(config_path.read_text())
        train_branch_final(b, config)


if __name__ == "__main__":
    main()
