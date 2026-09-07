import pandas as pd
import pytest

from developer.data_prepare.group_split import predefined_split_indices


def test_predefined_group_split_has_no_overlap():
    frame = pd.DataFrame({"group_id": ["a", "a", "b", "c"], "split": ["train", "train", "validation", "test"]})
    train, validation, test = predefined_split_indices(frame)
    assert train.tolist() == [0, 1]
    assert validation.tolist() == [2]
    assert test.tolist() == [3]


def test_cross_split_group_is_rejected():
    frame = pd.DataFrame({"group_id": ["same", "same", "other"], "split": ["train", "test", "validation"]})
    with pytest.raises(ValueError, match="数据泄漏"):
        predefined_split_indices(frame)


def test_every_label_must_exist_in_every_split():
    frame = pd.DataFrame({
        "group_id": ["a", "b", "c", "d"],
        "split": ["train", "validation", "test", "train"],
        "label": ["benign", "benign", "benign", "iodine"],
    })
    with pytest.raises(ValueError, match="标签没有覆盖"):
        predefined_split_indices(frame)
