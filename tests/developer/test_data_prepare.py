import json
from pathlib import Path

import pytest

from developer.data_prepare.build_manifest import build_manifest, infer_capture_provenance, infer_label
from developer.data_prepare.map_labels import normalize_label
from developer.data_prepare.split_dataset import (
    assign_group,
    split_manifest,
    stratified_group_assignments,
    validate_split_records,
)


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


def test_andmal_provenance_groups_malware_by_family():
    path = Path(
        "CIC-AndMal2017/extracted/Adware/Dowgin/"
        "06_14_2017-ad-dowgin-gdata-1c4e357a8ec5f13de4ffd57cc2711afe.pcap"
    )
    result = infer_capture_provenance(path)
    assert result["sample_md5"] == "1c4e357a8ec5f13de4ffd57cc2711afe"
    assert result["family"] == "Dowgin"
    assert result["group_id"] == "andmal:family:adware:dowgin"


def test_andmal_provenance_keeps_duplicate_benign_app_together():
    left = Path("CIC-AndMal2017/extracted/Benign/2015/08_04_2017-be-2015-OK-1-com.example.app.pcap")
    right = Path("CIC-AndMal2017/extracted/Benign/2016/07_08_2017-be-2016-com.example.app.pcap")
    assert infer_capture_provenance(left)["group_id"] == infer_capture_provenance(right)["group_id"]


def test_doh_malicious_provenance_groups_repeated_clients_and_times_by_scenario():
    base = "CIRA-CIC-DoHBrw-2020/extracted/PCAPs/DoHMalicious/"
    left = Path(base + "dsncat2_txt-tunnel_1111_doh1_2020-03-29T11_12_35.491984.pcap")
    right = Path(base + "dsncat2_txt-tunnel_1111_doh10_2020-03-29T13_29_46.933619.pcap")
    lmeta, rmeta = infer_capture_provenance(left), infer_capture_provenance(right)
    assert lmeta["tool"] == "dnscat2"
    assert lmeta["resolver"] == "cloudflare"
    assert lmeta["group_id"] == rmeta["group_id"]


def test_doh_benign_provenance_groups_by_browser_and_resolver():
    base = "CIRA-CIC-DoHBrw-2020/extracted/PCAPs/DoHBenign-NonDoH/AdGuard/"
    chrome = infer_capture_provenance(Path(base + "dump_00001_20200114102945.pcap"))
    firefox = infer_capture_provenance(Path(base + "1/dump.pcap"))
    assert chrome["group_id"] == "dohbrw:benign-scenario:chrome:adguard"
    assert firefox["group_id"] == "dohbrw:benign-scenario:firefox:adguard"
    assert chrome["group_id"] != firefox["group_id"]


def test_split_rejects_unverified_per_pcap_fallback():
    records = [
        {
            "path": f"A/benign/{index}.pcap", "dataset": "A", "label": "benign",
            "file_type": "pcap", "sha256": f"{index:064x}", "provenance_status": "pcap_only",
        }
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="拒绝退化"):
        stratified_group_assignments(records, ["sample_id", "sha256"])


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


def test_size_balancing_uses_capture_counts_not_only_group_counts():
    sizes = [50, 20, 10, 8, 6, 4, 1, 1]
    records = []
    for group_index, size in enumerate(sizes):
        records.extend({
            "path": f"A/dnscat2/{group_index}-{item}.pcap",
            "dataset": "A", "label": "dnscat2", "file_type": "pcap",
            "group_id": f"scenario-{group_index}", "provenance_status": "verified",
        } for item in range(size))
    assignments = stratified_group_assignments(records, ["sha256"])
    counts = {split: 0 for split in ("train", "validation", "test")}
    for record in records:
        group = f"dataset=A|group={record['group_id']}"
        counts[assignments[group]] += 1
    assert counts["train"] >= counts["test"] >= counts["validation"]
    assert set(counts.values()) != {50, 20, 30}


def test_split_manifest_writes_validation_report(tmp_path):
    records = [
        {
            "path": f"A/benign/{index}.pcap", "dataset": "A", "label": "benign",
            "file_type": "pcap", "group_id": f"app-{index}",
            "group_basis": "android_package", "provenance_status": "verified",
            "usable": True, "sha256": f"{index:064x}",
        }
        for index in range(10)
    ]
    source = tmp_path / "source.json"
    output = tmp_path / "split.json"
    source.write_text(json.dumps({"files": records}), encoding="utf-8")
    report = split_manifest(source, output)
    assert report["status"] == "passed"
    assert report["leaking_groups"] == 0
    assert (tmp_path / "split_report.json").exists()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["split_validation"]["usable_pcaps"] == 10


def test_validation_rejects_group_crossing_splits():
    records = [
        {"path": "a.pcap", "dataset": "A", "label": "benign", "file_type": "pcap", "usable": True, "group_id": "same", "split": "train"},
        {"path": "b.pcap", "dataset": "A", "label": "benign", "file_type": "pcap", "usable": True, "group_id": "same", "split": "test"},
    ]
    with pytest.raises(ValueError, match="跨 split"):
        validate_split_records(records, ["sha256"])
