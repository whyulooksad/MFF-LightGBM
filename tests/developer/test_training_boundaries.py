import json

import numpy as np
import pandas as pd
import pytest

from developer.detector.pca_baseline import reduce_feat_in_memory
from developer.detector.train_lightgbm import preprocess_detector_dataframe
from developer.representation.pretrain import load_rtd_texts


def test_rtd_holdout_is_group_based_and_deterministic(tmp_path):
    path = tmp_path / "pretrain.jsonl"
    rows = [
        {"text": f"group-{group}-flow-{flow}", "group_id": f"group-{group}", "label": group % 2}
        for group in range(10)
        for flow in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    train_a, holdout_a = load_rtd_texts(path)
    train_b, holdout_b = load_rtd_texts(path)
    assert (train_a, holdout_a) == (train_b, holdout_b)
    train_groups = {text.split("-flow-")[0] for text in train_a}
    holdout_groups = {text.split("-flow-")[0] for text in holdout_a}
    assert train_groups.isdisjoint(holdout_groups)


def test_pca_fit_is_unchanged_when_only_test_values_change():
    base = pd.DataFrame({
        "feat_0": [0.0, 1.0, 2.0, 100.0],
        "feat_1": [0.0, 2.0, 4.0, -100.0],
        "group_id": ["a", "b", "c", "d"],
        "split": ["train", "train", "validation", "test"],
    })
    changed = base.copy()
    changed.loc[3, ["feat_0", "feat_1"]] = [1_000_000.0, -1_000_000.0]
    left = reduce_feat_in_memory(base, n_components=1, fit_indices=[0, 1], verbose=False)
    right = reduce_feat_in_memory(changed, n_components=1, fit_indices=[0, 1], verbose=False)
    np.testing.assert_allclose(left.loc[:2, "feat_0"], right.loc[:2, "feat_0"])


def test_detector_rejects_ad_hoc_categorical_encoding():
    frame = pd.DataFrame({"label": [0, 1], "unexpected_text": ["a", "b"]})
    with pytest.raises(ValueError, match="禁止在全数据上"):
        preprocess_detector_dataframe(frame)
