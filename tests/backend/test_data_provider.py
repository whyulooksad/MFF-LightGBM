import json

import pandas as pd

from user_app.backend.data_provider import DataProvider


def _provider(tmp_path):
    task = tmp_path / "tasks" / "task-1"
    task.mkdir(parents=True)
    pd.DataFrame([{
        "flow_uid": "f1", "src_ip": "10.0.0.1", "src_port": 10,
        "dst_ip": "10.0.0.2", "dst_port": 443, "protocol": "tcp",
        "duration": 1.2, "total_packets": 3, "total_bytes": 250,
        "packet_time_offsets": "[0,0.2,1.2]", "packet_directions": "[1,-1,1]",
        "packet_lengths": "[60,120,70]",
    }]).to_csv(task / "flow_temporal.csv", index=False)
    pd.DataFrame([{
        "flow_uid": "f1", "connection_log": json.dumps({"service": "tls"}),
        "tls_log": json.dumps({"server_name": "example.test", "version": "TLSv1.3"}),
        "x509_log": "[]",
    }]).to_csv(task / "flow_features.csv", index=False)
    pd.DataFrame([{
        "flow_uid": "f1", "pred_label_name": "benign", "confidence": 0.9,
    }]).to_csv(task / "predictions.csv", index=False)
    (task / "summary.json").write_text(json.dumps({"total_flows": 1}), encoding="utf-8")
    return DataProvider(tmp_path)


def test_task_scoped_flows_and_detail(tmp_path):
    provider = _provider(tmp_path)
    result = provider.read_task_flows("task-1")
    assert result["total"] == 1
    assert result["rows"][0]["sni"] == "example.test"
    detail = provider.read_task_flow_detail("task-1", "f1")
    assert detail["tls"]["sni"] == "example.test"
    assert detail["temporal"]["packet_lengths"] == [60, 120, 70]


def test_task_predictions_and_summary(tmp_path):
    provider = _provider(tmp_path)
    assert provider.read_task_summary("task-1")["total_flows"] == 1
    assert provider.read_task_predictions("task-1")["rows"][0]["pred_label_name"] == "benign"


def test_task_path_cannot_escape_runtime(tmp_path):
    provider = _provider(tmp_path)
    assert provider.task_dir("../outside") is None
