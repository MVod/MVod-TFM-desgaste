# jordi/src/inference_utils.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple

SOURCES: Dict[str, Tuple[str, int]] = {
    "Normales (24)": ("normal", 0),
    "Desgastadas (24)": ("worn", 1),
    "Intermedias (36)": ("intermediate", 2),
}

_MODEL_SUFFIX: Dict[str, str] = {"A": ".pt", "B": ".pt", "C": ".pkl"}


def model_path(branch: str) -> Path:
    """Return the expected path for a branch's final trained model."""
    return Path(f"models/branch_{branch.lower()}_final{_MODEL_SUFFIX[branch]}")


def build_inference_index(
    source_dir: Path,
    label: int,
) -> Dict[str, Tuple[List[Path], int]]:
    """
    Scan source_dir for images matching Imagen_XXXXXX_R* and group by tool ID.
    Returns {tool_id: ([image_paths], label)}.
    tool_id format: "{number}_{label}" to avoid collisions across sources.
    """
    pattern = re.compile(r"^Imagen_(\d+)_R")
    index: Dict[str, Tuple[List[Path], int]] = {}
    for img_path in sorted(source_dir.iterdir()):
        if not img_path.is_file():
            continue
        m = pattern.match(img_path.name)
        if not m:
            continue
        tool_id = f"{m.group(1)}_{label}"
        if tool_id not in index:
            index[tool_id] = ([], label)
        index[tool_id][0].append(img_path)
    return index


def get_tool_ids(index: Dict[str, Tuple[List[Path], int]]) -> List[str]:
    """Return sorted list of tool IDs from an inference index."""
    return sorted(index.keys())
