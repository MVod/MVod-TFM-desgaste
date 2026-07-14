import re
from pathlib import Path
from typing import Dict, List, Tuple


def build_tool_index(
    data_dir: Path,
) -> Dict[str, Tuple[List[Path], int]]:
    """
    Scans data_dir/normal/ and data_dir/worn/ for images.
    Returns {tool_id: ([image_paths], label)} where label 0=normal, 1=worn.
    Image names must follow Imagen_XXXXXX_RYYY.* convention.
    """
    tool_index: Dict[str, Tuple[List[Path], int]] = {}
    pattern = re.compile(r"^Imagen_(\d+)_R")

    for label, subdir in ((0, "normal"), (1, "worn")):
        subpath = Path(data_dir) / subdir
        if not subpath.exists():
            continue
        for img_path in sorted(subpath.iterdir()):
            if not img_path.is_file():
                continue
            match = pattern.match(img_path.name)
            if not match:
                continue
            # Append label so normal/worn tools with the same number get distinct keys
            tool_id = f"{match.group(1)}_{label}"
            if tool_id not in tool_index:
                tool_index[tool_id] = ([], label)
            tool_index[tool_id][0].append(img_path)

    return tool_index


def generate_loocv_folds(
    tool_index: Dict[str, Tuple[List[Path], int]],
) -> List[Tuple[List[str], str]]:
    """
    Returns N (train_tool_ids, test_tool_id) tuples.
    Each tool appears exactly once as the test tool.
    """
    tool_ids = sorted(tool_index.keys())
    return [
        ([t for t in tool_ids if t != test_id], test_id)
        for test_id in tool_ids
    ]
