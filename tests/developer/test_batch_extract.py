import csv
import json

import pytest

from developer.data_prepare import batch_extract as module


def test_batch_extract_streams_combined_outputs_and_removes_temporary_files(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    pcap = prepared / "train" / "Dataset" / "benign" / "sample.pcap"
    pcap.parent.mkdir(parents=True)
    pcap.write_bytes(b"capture")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "path": "Dataset/extracted/sample.pcap",
        "prepared_path": "train/Dataset/benign/sample.pcap",
        "dataset": "Dataset", "label": "benign", "split": "train", "group_id": "app-1",
    }]}), encoding="utf-8")

    output_root = tmp_path / "per_capture"
    feature_all = output_root / "all_flow_features.csv"
    temporal_all = output_root / "all_flow_temporal.csv"
    monkeypatch.setattr(module, "extract_pcap", lambda _path, meta: (
        [{"label": meta["label"], "group_id": meta["group_id"], "split": meta["split"]}],
        [{"label": meta["label"], "group_id": meta["group_id"], "split": meta["split"]}],
    ))

    module.batch_extract(prepared, output_root, manifest)

    with feature_all.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["group_id"] == "app-1"
    assert not feature_all.with_suffix(".csv.tmp").exists()
    assert not temporal_all.with_suffix(".csv.tmp").exists()


def test_batch_extract_rejects_prepared_files_not_in_manifest(tmp_path):
    prepared = tmp_path / "prepared"
    extra = prepared / "train" / "Dataset" / "benign" / "stale.pcap"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"stale")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps({"files": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared 与 split_manifest 不一致"):
        module.batch_extract(prepared, tmp_path / "output", manifest)
