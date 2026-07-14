# jordi/tests/test_inference_utils.py
import cv2
import numpy as np
import pytest
from pathlib import Path
from inference_utils import (
    build_inference_index,
    get_tool_ids,
    model_path,
    SOURCES,
)


@pytest.fixture
def tmp_source_dir(tmp_path):
    """Fake directory: 2 tools, 3 images each, following Imagen_XXXXXX_RYY naming."""
    for tool_id, parts in [("000001", ["RM01", "RM02", "RM03"]),
                            ("000002", ["RM01", "RM02", "RM03"])]:
        for part in parts:
            img = np.ones((100, 200), dtype=np.uint8) * 240
            img[30:70, :] = 60
            cv2.imwrite(str(tmp_path / f"Imagen_{tool_id}_{part}.png"), img)
    return tmp_path


def test_build_inference_index_groups_by_tool(tmp_source_dir):
    index = build_inference_index(tmp_source_dir, label=0)
    assert set(index.keys()) == {"000001_0", "000002_0"}
    assert len(index["000001_0"][0]) == 3
    assert index["000001_0"][1] == 0


def test_build_inference_index_label_2(tmp_source_dir):
    index = build_inference_index(tmp_source_dir, label=2)
    assert all(k.endswith("_2") for k in index)
    assert all(v[1] == 2 for v in index.values())


def test_build_inference_index_empty_dir(tmp_path):
    assert build_inference_index(tmp_path, label=0) == {}


def test_get_tool_ids_sorted(tmp_source_dir):
    index = build_inference_index(tmp_source_dir, label=1)
    ids = get_tool_ids(index)
    assert ids == sorted(ids)


def test_model_path_extensions():
    assert model_path("A").suffix == ".pt"
    assert model_path("B").suffix == ".pt"
    assert model_path("C").suffix == ".pkl"


def test_model_path_names():
    assert model_path("A") == Path("models/branch_a_final.pt")
    assert model_path("B") == Path("models/branch_b_final.pt")
    assert model_path("C") == Path("models/branch_c_final.pkl")


def test_sources_keys():
    assert set(SOURCES.keys()) == {"Normales (24)", "Desgastadas (24)", "Intermedias (36)"}
    assert SOURCES["Normales (24)"] == ("normal", 0)
    assert SOURCES["Desgastadas (24)"] == ("worn", 1)
    assert SOURCES["Intermedias (36)"] == ("intermediate", 2)
