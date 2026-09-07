"""Batch-call the exact same extractor used by production inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from user_app.inference.extract_flow_features import CSV_HEADERS, TEMPORAL_HEADERS, extract_pcap, write_csv
from user_app.inference.contract import CLASS_LABELS
from developer.config import FLOW_FEATURES_ALL_CSV, FLOW_METADATA_TEMPORAL_ALL_CSV


def _manifest_records(manifest_path: Path | None) -> dict[str, dict]:
    if manifest_path is None:
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for record in data.get("files", []):
        for key in (record.get("path"), record.get("prepared_path")):
            if key:
                records[str(key).replace("\\", "/")] = record
    return records


def _infer_label(relative: Path) -> str:
    matches = [part.lower() for part in relative.parts if part.lower() in CLASS_LABELS]
    if len(matches) != 1:
        raise ValueError(f"无法从路径唯一确定八分类标签: {relative}；请在 manifest 记录中填写 label")
    return matches[0]


def batch_extract(prepared_root: Path, output_root: Path, manifest_path: Path | None = None) -> None:
    records = _manifest_records(manifest_path)
    all_rows, all_temporal = [], []
    for pcap in sorted(path for path in prepared_root.rglob("*") if path.suffix.lower() in {".pcap", ".pcapng"}):
        relative = pcap.relative_to(prepared_root)
        record = records.get(relative.as_posix(), {})
        target = output_root / relative.parent / pcap.stem
        target.mkdir(parents=True, exist_ok=True)
        split = str(record.get("split") or (relative.parts[0] if relative.parts and relative.parts[0] in {"train", "validation", "test"} else ""))
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{relative} 缺少合法 split；先运行 split_dataset.py")
        rows, temporal = extract_pcap(
            pcap,
            {
                "group_id": record.get("group_id") or relative.as_posix(),
                "split": split,
                "label": record.get("label") or _infer_label(relative),
                "dataset_source": record.get("dataset") or (relative.parts[1] if len(relative.parts) > 1 else ""),
                "subfolder": str(relative.parent),
            },
        )
        write_csv(target / "flow_features.csv", rows, CSV_HEADERS)
        write_csv(target / "flow_temporal.csv", temporal, TEMPORAL_HEADERS)
        all_rows.extend(rows)
        all_temporal.extend(temporal)
    write_csv(FLOW_FEATURES_ALL_CSV, all_rows, CSV_HEADERS)
    write_csv(FLOW_METADATA_TEMPORAL_ALL_CSV, all_temporal, TEMPORAL_HEADERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    batch_extract(args.prepared_root, args.output_root, args.manifest)
