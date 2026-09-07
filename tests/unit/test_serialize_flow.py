import json

import pandas as pd

from user_app.inference.contract import FEATURE_SCHEMA_VERSION, NEW_FORMAT_NUM_FEATURES
from user_app.inference.serialize_flow import preprocess


def test_preprocess_preserves_json_quotes_and_builds_text_events(tmp_path):
    row = {
        "flow_uid": "flow-1",
        "src_ip": "192.0.2.10",
        "src_port": 50000,
        "dst_ip": "198.51.100.20",
        "dst_port": 443,
        "protocol": "tcp",
        "timestamp": 1.25,
        "label": "benign",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "connection_log": json.dumps(
            {"proto": "tcp", "duration": 1.25, "orig_pkts": 3, "resp_pkts": 2}
        ),
        "tls_log": json.dumps(
            {"version": "TLSv1.3", "server_name": "example.test", "established": True}
        ),
        "x509_log": json.dumps(
            [{"certificate.subject": "CN=example.test", "san.dns": ["example.test"]}]
        ),
        "group_id": "capture-1",
        "split": "train",
        "pcap_filename": "capture.pcap",
        "dataset_source": "fixture",
    }
    row.update({column: 0.0 for column in NEW_FORMAT_NUM_FEATURES})

    source = tmp_path / "flow_features.csv"
    output = tmp_path / "flows.jsonl"
    pd.DataFrame([row]).to_csv(source, index=False)

    flows = preprocess(str(source), str(output), target="feature")

    assert len(flows) == 1
    assert flows[0]["num_events"] == 3
    assert '"t":"c"' in flows[0]["text"]
    assert '"t":"s"' in flows[0]["text"]
    assert '"t":"x"' in flows[0]["text"]
    assert '"sni":"example.test"' in flows[0]["text"]
    assert json.loads(output.read_text(encoding="utf-8"))["text"] == flows[0]["text"]
