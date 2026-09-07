import json
from pathlib import Path

import pytest

from developer.data_prepare.build_manifest import build_manifest, infer_label
from developer.data_prepare.map_labels import normalize_label
from developer.data_prepare.split_dataset import assign_group, stratified_group_assignments


def test_manifest_hashes_source_files(tmp_path):
    (tmp_path / "sample.pcap").write_bytes(b"pcap")
    manifest = build_manifest(tmp_path)
    assert manifest["files"][0]["path"] == "sample.pcap"
    assert len(manifest["files"][0]["sha256"]) == 64


def test_manifest_excludes_empty_capture_without_deleting_it(tmp_path):
    capture = tmp_path / "empty.pcap"
    capture.write_bytes(b"")
    record = build_manifest(tmp_path)["files"][0]
    assert capture.exists()
    assert record["usable"] is False
    assert record["exclusion_reason"] == "empty_capture"


def test_manifest_applies_explicit_capture_provenance(tmp_path):
    (tmp_path / "Dataset" / "benign").mkdir(parents=True)
    relative = "Dataset/benign/capture.pcap"
    (tmp_path / relative).write_bytes(b"pcap")
    manifest = build_manifest(tmp_path, {relative: {"capture_id": "capture-42", "device": "phone-a"}})
    record = manifest["files"][0]
    assert record["capture_id"] == "capture-42"
    assert record["device"] == "phone-a"
    assert record["provenance_status"] == "verified"


def test_explicit_group_id_overrides_automatic_group_fields():
    records = []
    for index in range(3):
        records.append({
            "path": f"A/benign/{index}.pcap", "dataset": "A", "label": "benign",
            "file_type": "pcap", "group_id": f"experiment-{index}",
            "sample_id": f"sample-{index}", "sha256": f"{index:064x}",
        })
    assignments = stratified_group_assignments(records, ["sample_id", "sha256"])
    assert set(assignments) == {f"dataset=A|group=experiment-{index}" for index in range(3)}
    records[0]["group_id"] = "dataset=A|group=experiment-0"
    assert stratified_group_assignments(records, ["sample_id", "sha256"])


def test_label_mapping_rejects_unknown_labels():
    assert normalize_label("Normal") == "benign"
    with pytest.raises(ValueError):
        normalize_label("unmapped-family")


def test_official_dohbrw_dsncat2_filename_maps_to_dnscat2():
    path = Path("CIRA-CIC-DoHBrw-2020/extracted/PCAPs/DoHMalicious/dsncat2_default-baseline.pcap")
    assert infer_label(path) == "dnscat2"


def test_group_split_is_deterministic():
    assert assign_group("same-apk") == assign_group("same-apk")


def test_stratified_group_split_keeps_every_dataset_label_in_all_splits():
    records = []
    for dataset in ("A", "B"):
        for label in ("benign", "adware"):
            for index in range(10):
                records.append({
                    "path": f"{dataset}/{label}/{index}.pcap",
                    "dataset": dataset,
                    "label": label,
                    "file_type": "pcap",
                    "sample_id": f"{label}-{index}",
                    "sha256": f"{index:064x}",
                })
    assignments = stratified_group_assignments(records, ["sample_id", "sha256"])
    for dataset in ("A", "B"):
        for label in ("benign", "adware"):
            splits = {
                assignments[f"dataset={dataset}|sample_id={label}-{index}"]
                for index in range(10)
            }
            assert splits == {"train", "validation", "test"}


def test_stratified_group_split_rejects_class_with_fewer_than_three_groups():
    records = [
        {
            "path": f"A/iodine/{index}.pcap", "dataset": "A", "label": "iodine",
            "file_type": "pcap", "sample_id": f"sample-{index}", "sha256": f"{index:064x}",
        }
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="至少需要 3 个"):
        stratified_group_assignments(records, ["sample_id", "sha256"])
