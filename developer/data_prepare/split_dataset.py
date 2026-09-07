"""Create a deterministic, group-safe train/validation/test split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from itertools import permutations
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


def _balanced_group_assignment(
    group_sizes: dict[str, int],
    ratios: dict[str, float],
    stratum: tuple[str, str],
) -> dict[str, str]:
    """Keep groups intact while balancing the number of capture files.

    Group sizes vary substantially in DoHBrw. Allocating only the same number
    of groups can therefore create a badly skewed record split. The three
    largest groups are seeded into distinct splits using the best permutation;
    remaining groups greedily minimise squared distance from target PCAP counts.
    """
    split_names = list(ratios)
    if len(group_sizes) < len(split_names):
        raise ValueError(f"至少需要 {len(split_names)} 个独立来源组，实际只有 {len(group_sizes)} 个")

    stable_hash = lambda group: hashlib.sha256(
        f"{stratum[0]}|{stratum[1]}|{group}".encode("utf-8")
    ).hexdigest()
    ordered = sorted(group_sizes, key=lambda group: (-group_sizes[group], stable_hash(group)))
    total = sum(group_sizes.values())
    targets = {name: total * ratio for name, ratio in ratios.items()}

    seed_groups = ordered[: len(split_names)]
    best_order = min(
        permutations(split_names),
        key=lambda assignment: (
            sum((group_sizes[group] - targets[split]) ** 2 for group, split in zip(seed_groups, assignment)),
            assignment,
        ),
    )
    result = {group: split for group, split in zip(seed_groups, best_order)}
    totals = {name: 0 for name in split_names}
    for group, split in result.items():
        totals[split] += group_sizes[group]

    for group in ordered[len(split_names) :]:
        size = group_sizes[group]
        split = min(
            split_names,
            key=lambda name: (
                (totals[name] + size - targets[name]) ** 2 - (totals[name] - targets[name]) ** 2,
                split_names.index(name),
            ),
        )
        result[group] = split
        totals[split] += size
    return result


def stratified_group_assignments(
    records: list[dict],
    group_fields: list[str],
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Balance immutable groups independently inside every dataset/label stratum."""
    ratios = ratios or SPLIT_RATIOS
    strata: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_owner: dict[str, tuple[str, str]] = {}
    for record in records:
        if record.get("file_type") not in VALID_CAPTURE_TYPES or not record.get("usable", True):
            continue
        dataset = str(record.get("dataset", "")).strip()
        label = str(record.get("label", "")).strip().lower()
        if not dataset or label not in CLASS_LABELS:
            raise ValueError(f"PCAP 缺少合法 dataset/label: {record.get('path')}")
        if record.get("provenance_status") == "pcap_only":
            raise ValueError(
                f"PCAP 尚未确认来源分组，拒绝退化为单文件随机划分: {record.get('path')}"
            )
        group = record_group_id(record, group_fields)
        owner = (dataset, label)
        if group in group_owner and group_owner[group] != owner:
            raise ValueError(f"同一来源组跨越了不同数据集或标签: {group}")
        group_owner[group] = owner
        strata[owner][group] += 1

    assignments: dict[str, str] = {}
    for stratum, group_sizes in sorted(strata.items()):
        assignments.update(_balanced_group_assignment(dict(group_sizes), ratios, stratum))
    return assignments


def validate_split_records(records: list[dict], group_fields: list[str]) -> dict:
    """Fail on leakage/incomplete strata and return an audit-friendly report."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    strata: dict[tuple[str, str], dict[str, dict[str, set | int]]] = defaultdict(
        lambda: defaultdict(lambda: {"pcaps": 0, "groups": set()})
    )
    usable_count = 0
    excluded_count = 0
    for record in records:
        if record.get("file_type") not in VALID_CAPTURE_TYPES:
            continue
        if not record.get("usable", True):
            excluded_count += 1
            continue
        usable_count += 1
        split = str(record.get("split", ""))
        if split not in SPLIT_RATIOS:
            raise ValueError(f"PCAP 缺少合法 split: {record.get('path')}")
        group = record_group_id(record, group_fields)
        group_splits[group].add(split)
        key = (str(record.get("dataset", "")), str(record.get("label", "")))
        strata[key][split]["pcaps"] += 1
        strata[key][split]["groups"].add(group)

    leaking = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaking:
        raise ValueError(f"发现来源组跨 split 泄漏: {leaking[:5]}")

    report_strata = []
    for (dataset, label), split_data in sorted(strata.items()):
        missing = [name for name in SPLIT_RATIOS if not split_data[name]["groups"]]
        if missing:
            raise ValueError(f"{dataset}/{label} 缺少 split: {missing}")
        total_pcaps = sum(int(split_data[name]["pcaps"]) for name in SPLIT_RATIOS)
        report_strata.append({
            "dataset": dataset,
            "label": label,
            "total_pcaps": total_pcaps,
            "total_groups": len(set().union(*(split_data[name]["groups"] for name in SPLIT_RATIOS))),
            "splits": {
                name: {
                    "pcaps": split_data[name]["pcaps"],
                    "groups": len(split_data[name]["groups"]),
                    "pcap_ratio": round(int(split_data[name]["pcaps"]) / total_pcaps, 6),
                }
                for name in SPLIT_RATIOS
            },
        })
    return {
        "status": "passed",
        "usable_pcaps": usable_count,
        "excluded_pcaps": excluded_count,
        "leaking_groups": 0,
        "strata": report_strata,
    }


def split_manifest(source: Path, output: Path, group_fields: list[str] | None = None) -> dict:
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
        "method": "dataset-label-stratified-size-balanced-group",
        "ratios": SPLIT_RATIOS,
        "group_fields": group_fields,
    }
    report = validate_split_records(records, group_fields)
    manifest["split_validation"] = report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output.with_name("split_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


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
        if target.exists() and not os.path.samefile(source, target):
            raise FileExistsError(f"prepared 目标已存在但不指向清单中的源文件: {target}")
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
    report = split_manifest(args.source, args.output, args.group_fields)
    print(f"validation={report['status']} usable_pcaps={report['usable_pcaps']} report={args.output.with_name('split_report.json')}")
    if args.materialize_root:
        print(f"materialized={materialize_pcap_links(args.output, args.materialize_root)}")
