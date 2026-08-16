# app/tests/test_data_provider.py
import os
import pytest
from data_provider import DataProvider

@pytest.fixture
def provider():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return DataProvider(
        flow_features_dir=os.path.join(root, "data", "flow_features"),
        pipeline_output_dir=os.path.join(root, "data", "pipeline", "output"),
        runtime_dir=os.path.join(root, "data", "web_runtime"),
    )

def test_read_metadata_pagination(provider):
    rows = provider.read_metadata(limit=5, offset=0)
    assert len(rows) == 5
    assert "flow_uid" in rows[0]
    assert "src_ip" in rows[0]

def test_read_metadata_detail(provider):
    rows = provider.read_metadata(limit=1, offset=0)
    detail = provider.read_metadata_detail(rows[0]["flow_uid"])
    assert detail is not None
    assert "metadata" in detail
    assert "tls" in detail
    assert "temporal" in detail

def test_read_predictions_static(provider):
    out = provider.read_predictions(source="static", limit=10, offset=0)
    assert out["total"] > 0
    assert len(out["rows"]) == 10
    assert "pred_label_name" in out["rows"][0]

def test_read_evaluation_static(provider):
    out = provider.read_evaluation(source="static")
    assert "algos" in out
    names = [a["name"] for a in out["algos"]]
    assert "lightgbm" in names
    assert out["test_set"]["samples"] > 0
