import numpy as np
import pandas as pd
import pytest

from pipeline.supcon_ae import SupConAEReducer, replace_semantic_features


@pytest.fixture
def fitted_reducer(tmp_path):
    rng = np.random.default_rng(42)
    train_labels = np.repeat(np.arange(3), 20)
    val_labels = np.repeat(np.arange(3), 6)
    train = rng.normal(size=(60, 12)).astype(np.float32)
    train += train_labels[:, None] * 0.4
    val = rng.normal(size=(18, 12)).astype(np.float32)
    val += val_labels[:, None] * 0.4
    columns = [f"feat_{index}" for index in range(12)]
    checkpoint = tmp_path / "supcon.pt"
    reducer = SupConAEReducer(
        config={
            "latent_dim": 4,
            "hidden_dims": [16, 8],
            "proj_dim": 4,
            "batch_size": 16,
            "epochs": 2,
            "early_stopping_patience": 2,
            "balance_classes": False,
            "seed": 7,
        },
        device="cpu",
        verbose=False,
    )
    reducer.fit(train, train_labels, val, val_labels, columns, checkpoint)
    return reducer, checkpoint, train, columns


def test_fit_transform_save_and_load(fitted_reducer):
    reducer, checkpoint, train, columns = fitted_reducer
    expected = reducer.transform(train, columns)
    assert expected.shape == (60, 4)
    assert checkpoint.exists()

    loaded = SupConAEReducer.load(checkpoint, device="cpu", verbose=False)
    actual = loaded.transform(train, columns)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert loaded.feature_columns == columns
    assert loaded.classes_.tolist() == [0, 1, 2]


def test_checkpoint_rejects_wrong_feature_order(fitted_reducer):
    reducer, _, train, columns = fitted_reducer
    with pytest.raises(ValueError, match="columns/order"):
        reducer.transform(train, list(reversed(columns)))


def test_replace_semantic_features_preserves_manual_features(fitted_reducer):
    reducer, _, train, columns = fitted_reducer
    dataframe = pd.DataFrame(train, columns=columns)
    dataframe.insert(0, "flow_uid", [f"flow-{index}" for index in range(len(dataframe))])
    dataframe["bytes_total"] = np.arange(len(dataframe))
    dataframe["label"] = np.repeat(np.arange(3), 20)

    reduced = replace_semantic_features(dataframe, reducer)
    assert [column for column in reduced if column.startswith("feat_")] == [
        "feat_0", "feat_1", "feat_2", "feat_3"
    ]
    assert reduced["flow_uid"].tolist() == dataframe["flow_uid"].tolist()
    assert reduced["bytes_total"].tolist() == dataframe["bytes_total"].tolist()


def test_legacy_checkpoint_is_rejected(tmp_path):
    import torch

    path = tmp_path / "legacy.pt"
    torch.save({"input_dim": 12, "cfg": {}}, path)
    with pytest.raises(ValueError, match="Legacy"):
        SupConAEReducer.load(path, device="cpu", verbose=False)
