import pytest
from evaluation.loocv import build_tool_index, generate_loocv_folds


def test_build_tool_index_counts(tmp_data_dir):
    index = build_tool_index(tmp_data_dir)
    assert len(index) == 4  # 2 normal + 2 worn tools


def test_build_tool_index_labels(tmp_data_dir):
    index = build_tool_index(tmp_data_dir)
    assert index["000001_0"][1] == 0  # normal
    assert index["000003_1"][1] == 1  # worn


def test_build_tool_index_image_count(tmp_data_dir):
    index = build_tool_index(tmp_data_dir)
    for tool_id, (paths, label) in index.items():
        assert len(paths) == 3


def test_generate_loocv_folds_count(tmp_data_dir):
    index = build_tool_index(tmp_data_dir)
    folds = generate_loocv_folds(index)
    assert len(folds) == 4  # one fold per tool


def test_generate_loocv_folds_no_leakage(tmp_data_dir):
    index = build_tool_index(tmp_data_dir)
    folds = generate_loocv_folds(index)
    for train_ids, test_id in folds:
        assert test_id not in train_ids
        assert len(train_ids) == 3  # all tools minus the test one


def test_generate_loocv_folds_all_tools_appear_as_test(tmp_data_dir):
    index = build_tool_index(tmp_data_dir)
    folds = generate_loocv_folds(index)
    test_ids = {test_id for _, test_id in folds}
    assert test_ids == set(index.keys())
