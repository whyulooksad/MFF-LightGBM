"""Batch-call the exact same extractor used by production inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from user_app.inference.extract_flow_features import CSV_HEADERS, TEMPORAL_HEADERS, extract_pcap, write_csv
from developer.config import DATASET_SPLIT_MANIFEST


def _manifest_records(manifest_path: Path) -> dict[str, dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for record in data.get("files", []):
        for key in (record.get("path"), record.get("prepared_path")):
            if key:
                records[str(key).replace("\\", "/")] = record
    return records


def batch_extract(prepared_root: Path, output_root: Path, manifest_path: Path = DATASET_SPLIT_MANIFEST) -> None:
    records = _manifest_records(manifest_path)
    pcaps = sorted(path for path in prepared_root.rglob("*") if path.suffix.lower() in {".pcap", ".pcapng"})
    unknown = [pcap.relative_to(prepared_root).as_posix() for pcap in pcaps if pcap.relative_to(prepared_root).as_posix() not in records]
    expected = {
        str(record["prepared_path"]).replace("\\", "/")
        for record in json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
        if record.get("file_type") in {"pcap", "pcapng"} and record.get("usable", True)
    }
    actual = {pcap.relative_to(prepared_root).as_posix() for pcap in pcaps}
    missing = sorted(expected - actual)
    if unknown or missing:
        raise ValueError(
            "prepared 与 split_manifest 不一致；请重新生成 prepared。"
            f" 多余={unknown[:5]} 缺少={missing[:5]}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    feature_all = output_root / "all_flow_features.csv"
    temporal_all = output_root / "all_flow_temporal.csv"
    feature_tmp = feature_all.with_suffix(feature_all.suffix + ".tmp")
    temporal_tmp = temporal_all.with_suffix(temporal_all.suffix + ".tmp")
    try:
        with feature_tmp.open("w", newline="", encoding="utf-8") as feature_stream, temporal_tmp.open(
            "w", newline="", encoding="utf-8"
        ) as temporal_stream:
            feature_writer = csv.DictWriter(feature_stream, fieldnames=CSV_HEADERS, extrasaction="ignore")
            temporal_writer = csv.DictWriter(temporal_stream, fieldnames=TEMPORAL_HEADERS, extrasaction="ignore")
            feature_writer.writeheader()
            temporal_writer.writeheader()
            for pcap in pcaps:
                relative = pcap.relative_to(prepared_root)
                record = records[relative.as_posix()]
                target = output_root / relative.parent / pcap.stem
                target.mkdir(parents=True, exist_ok=True)
                split = str(record.get("split") or (relative.parts[0] if relative.parts and relative.parts[0] in {"train", "validation", "test"} else ""))
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"{relative} 缺少合法 split；先运行 split_dataset.py")
                rows, temporal = extract_pcap(
                    pcap,
                    {
                        "group_id": record["group_id"],
                        "split": split,
                        "label": record["label"],
                        "dataset_source": record["dataset"],
                        "subfolder": str(relative.parent),
                    },
                )
                write_csv(target / "flow_features.csv", rows, CSV_HEADERS)
                write_csv(target / "flow_temporal.csv", temporal, TEMPORAL_HEADERS)
                # Keep only one capture's flows in memory; the combined files can
                # be much larger than RAM for the two official datasets.
                feature_writer.writerows(rows)
                temporal_writer.writerows(temporal)
        os.replace(feature_tmp, feature_all)
        os.replace(temporal_tmp, temporal_all)
    finally:
        feature_tmp.unlink(missing_ok=True)
        temporal_tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DATASET_SPLIT_MANIFEST)
    args = parser.parse_args()
    batch_extract(args.prepared_root, args.output_root, args.manifest)
