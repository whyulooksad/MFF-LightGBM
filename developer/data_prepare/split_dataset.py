"""Create a deterministic, group-safe train/validation/test split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from user_app.inference.contract import CLASS_LABELS


SPLIT_RATIOS = {"train": 0.70, "validation": 0.10, "test": 0.20}
VALID_CAPTURE_TYPES = {"pcap", "pcapng"}


def assign_group(group: str, train: float = 0.7, validation: float = 0.1) -> str:
    value = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def record_group_id(record: dict, group_fields: list[str]) -> str:
    explicit_group = str(record.get("group_id", "")).strip()
    if explicit_group:
        if explicit_group.startswith("dataset="):
            return explicit_group
        dataset = str(record.get("dataset", "")).strip()
        return "|".join(part for part in (f"dataset={dataset}" if dataset else "", f"group={explicit_group}") if part)
    populated = [
        (field, str(record.get(field, "")).strip())
        for field in group_fields
        if str(record.get(field, "")).strip()
    ]
    if populated:
        # Fields are ordered strongest to weakest.  Use exactly one unit:
        # e.g. all captures from the same APK share sample_id; otherwise a
        # capture id or finally the PCAP hash keeps all its flows together.
        field, value = populated[0]
        dataset = str(record.get("dataset", "")).strip()
        return "|".join(part for part in (f"dataset={dataset}" if dataset else "", f"{field}={value}") if part)
    source_path = str(record.get("path", "")).strip()
    if source_path:
        return f"pcap={source_path}"
    raise ValueError(f"记录缺少分组来源 {group_fields} 和 path: {record}")


def _allocation_counts(group_count: int, ratios: dict[str, float]) -> dict[str, int]:
    """Allocate whole groups by ratio while keeping every split non-empty."""
    if group_count < len(ratios):
        raise ValueError(f"至少需要 {len(ratios)} 个独立来源组，实际只有 {group_count} 个")
    raw = {name: group_count * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = group_count - sum(counts.values())
    order = sorted(ratios, key=lambda name: (raw[name] - counts[name], ratios[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    for empty in [name for name, count in counts.items() if count == 0]:
        donors = [name for name, count in counts.items() if count > 1]
        if not donors:
            raise ValueError(f"无法让所有集合非空: groups={group_count}, counts={counts}")
        donor = max(donors, key=lambda name: (counts[name] - raw[name], counts[name]))
        counts[donor] -= 1
        counts[empty] += 1
    return counts


def stratified_group_assignments(
    records: list[dict],
    group_fields: list[str],
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Balance immutable groups independently inside every dataset/label stratum."""
    ratios = ratios or SPLIT_RATIOS
    strata: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_owner: dict[str, tuple[str, str]] = {}
    for record in records:
        if record.get("file_type") not in VALID_CAPTURE_TYPES or not record.get("usable", True):
            continue
        dataset = str(record.get("dataset", "")).strip()
        label = str(record.get("label", "")).strip().lower()
        if not dataset or label not in CLASS_LABELS:
            raise ValueError(f"PCAP 缺少合法 dataset/label: {record.get('path')}")
        group = record_group_id(record, group_fields)
        owner = (dataset, label)
        if group in group_owner and group_owner[group] != owner:
            raise ValueError(f"同一来源组跨越了不同数据集或标签: {group}")
        group_owner[group] = owner
        strata[owner].add(group)

    assignments: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        counts = _allocation_counts(len(groups), ratios)
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha256(f"{stratum[0]}|{stratum[1]}|{group}".encode("utf-8")).hexdigest(),
        )
        cursor = 0
        for split_name in ratios:
            next_cursor = cursor + counts[split_name]
            for group in ordered[cursor:next_cursor]:
                assignments[group] = split_name
            cursor = next_cursor
    return assignments


def split_manifest(source: Path, output: Path, group_fields: list[str] | None = None) -> None:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    group_fields = group_fields or ["sample_id", "capture_id", "apk_sha256", "sha256"]
    records = manifest.get("files", [])
    assignments = stratified_group_assignments(records, group_fields)
    for record in records:
        if record.get("file_type") not in VALID_CAPTURE_TYPES or not record.get("usable", True):
            continue
        group = record_group_id(record, group_fields)
        record["group_id"] = group
        record["group_basis"] = str(record.get("group_basis") or "").strip() or next(
            (field for field in group_fields if str(record.get(field, "")).strip()), "path"
        )
        record["split"] = assignments[group]
    manifest["split_policy"] = {
        "method": "dataset-label-stratified-group",
        "ratios": SPLIT_RATIOS,
        "group_fields": group_fields,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def materialize_pcap_links(manifest_path: Path, prepared_root: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    count = 0
    for record in manifest.get("files", []):
        if record.get("file_type") not in {"pcap", "pcapng"} or not record.get("usable", True):
            continue
        if not record.get("label") or record.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"PCAP 缺少 label/split: {record.get('path')}")
        source = source_root / record["path"]
        name = f"{record['sha256'][:12]}_{source.name}"
        target = prepared_root / record["split"] / record.get("dataset", "unknown") / record["label"] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)  # same-volume hardlink: no duplicate dataset storage
        record["prepared_path"] = target.relative_to(prepared_root).as_posix()
        count += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--group-fields", nargs="+", default=["sample_id", "capture_id", "apk_sha256", "sha256"])
    parser.add_argument("--materialize-root", type=Path, default=None, help="可选：按 split/dataset/label 建立 PCAP 硬链接")
    args = parser.parse_args()
    split_manifest(args.source, args.output, args.group_fields)
    if args.materialize_root:
        print(f"materialized={materialize_pcap_links(args.output, args.materialize_root)}")
