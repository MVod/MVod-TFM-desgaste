"""
pipeline.py — Single entry point for all branch runs.

Usage:
    uv run pipeline.py --branch A --config configs/branch_a.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script (not via pytest)
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List

from evaluation.loocv import build_tool_index, generate_loocv_folds
from evaluation.metrics import compute_metrics, aggregate_fold_metrics
from tracking.mlflow_logger import RunLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_branch(branch: str):
    if branch == "A":
        from branches.branch_a_mil import BranchA
        return BranchA()
    if branch == "B":
        from branches.branch_e_unet import BranchE
        return BranchE()
    if branch == "C":
        from branches.branch_c_profile import BranchC
        return BranchC()
    raise ValueError(f"Unknown branch: {branch!r}. Must be one of A, B, C.")


def run(branch_name: str, config: Dict) -> None:
    set_seed(config["seed"])
    data_dir = Path(config["data_dir"])
    model_dir = Path(config.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    save_models = config.get("save_models", True)

    tool_index = build_tool_index(data_dir)
    folds = generate_loocv_folds(tool_index)

    logger = RunLogger(
        experiment_name=f"branch_{branch_name}",
        tracking_uri=config.get("mlflow_uri", "./mlruns"),
    )

    branch = load_branch(branch_name)

    fold_metrics = []
    all_true: List[int] = []
    all_scores: List[float] = []

    for fold_id, (train_ids, test_id) in enumerate(folds):
        print(f"\n[Branch {branch_name}] Fold {fold_id + 1}/{len(folds)} — test tool: {test_id}")
        branch.train(train_ids, tool_index, config)
        if save_models:
            model_path = model_dir / f"branch_{branch_name}_fold{fold_id:02d}.pt"
            branch.save(model_path)
        else:
            model_path = None
        tool_score, _ = branch.predict(test_id, tool_index, config)
        true_label = tool_index[test_id][1]
        metrics = compute_metrics([true_label], [tool_score])
        metrics["tool_score"] = tool_score
        metrics["true_label"] = true_label
        fold_metrics.append(metrics)
        all_true.append(true_label)
        all_scores.append(tool_score)
        print(f"  tool_score={tool_score:.3f} | true={true_label} | F1={metrics['f1']:.2f}")
        logger.log_fold(
            fold_id=fold_id,
            test_tool_id=test_id,
            params={**config, "branch": branch_name},
            metrics=metrics,
            model_path=model_path,
        )

    summary = aggregate_fold_metrics(fold_metrics, all_true=all_true, all_scores=all_scores)
    logger.log_summary(fold_metrics=fold_metrics, params={**config, "branch": branch_name}, all_true=all_true, all_scores=all_scores)
    print(f"\n=== Branch {branch_name} LOO-CV Summary ===")
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v:.4f}")


def run_viz(branch_name: str, config: Dict) -> None:
    """Train on all data, predict on all tools with visualizations enabled."""
    set_seed(config["seed"])
    data_dir = Path(config["data_dir"])
    vis_config = {**config, "vis_dir": "output/vis"}

    tool_index = build_tool_index(data_dir)
    all_tool_ids = list(tool_index.keys())

    branch = load_branch(branch_name)

    print(f"\n[Branch {branch_name}] Viz pass — training on {len(all_tool_ids)} tools...")
    branch.train(all_tool_ids, tool_index, vis_config)

    # Pick 2 normal + 2 worn tools for visualization (first and last of each class)
    normals = sorted([t for t in all_tool_ids if tool_index[t][1] == 0])
    worns = sorted([t for t in all_tool_ids if tool_index[t][1] == 1])
    viz_tools = normals[:1] + normals[-1:] + worns[:1] + worns[-1:]

    print(f"[Branch {branch_name}] Generating visualizations for: {viz_tools}")
    for tool_id in viz_tools:
        score, _ = branch.predict(tool_id, tool_index, vis_config)
        label = tool_index[tool_id][1]
        print(f"  {tool_id} | score={score:.3f} | true={label}")
    print(f"[Branch {branch_name}] Visualizations saved to output/vis/{branch_name}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wear classification pipeline")
    parser.add_argument("--branch", required=True, choices=["A", "B", "C"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--viz", action="store_true",
                        help="Skip LOO-CV; train on all data and generate visualizations")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.viz:
        run_viz(branch_name=args.branch, config=config)
    else:
        run(branch_name=args.branch, config=config)


if __name__ == "__main__":
    main()
