"""Build a reproducible inventory with sizes and SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from developer.config import DATASET_MANIFESTS_DIR, DATASET_SOURCE_DIR
from user_app.inference.contract import CLASS_LABELS


PROVENANCE_FIELDS = (
    "group_id", "group_basis", "sample_id", "capture_id", "apk_sha256",
    "sample_md5", "family", "device", "capture_batch", "tool", "browser",
    "resolver", "scenario",
)

CAPTURE_SUFFIXES = {".pcap", ".pcapng"}
INVENTORY_SUFFIXES = {*CAPTURE_SUFFIXES, ".csv"}
DOH_RESOLVERS = {
    "1111": "cloudflare",
    "99911": "quad9",
    "dnsadguardcom": "adguard",
    "dnsgoogle": "google",
}
DOH_MALICIOUS_RE = re.compile(
    r"^(?P<tool>dns2tcp|dsncat2|iodine)_(?P<configuration>.+)_"
    r"(?P<resolver>1111|99911|dnsadguardcom|dnsgoogle)_doh(?P<client>\d+)_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T.+)\.(?:pcap|pcapng)$",
    re.IGNORECASE,
)
ANDMAL_HASH_RE = re.compile(r"-([0-9a-f]{32})\.(?:pcap|pcapng)$", re.IGNORECASE)
ANDMAL_BENIGN_PREFIX_RE = re.compile(r"^\d{2}_\d{2}_\d{4}-be-\d{4}-(?:OK-\d+-)?", re.IGNORECASE)


def infer_label(relative: Path) -> str | None:
    normalized = relative.as_posix().lower().replace("_", " ").replace("-", " ")
    # CIRA-CIC-DoHBrw-2020 spells dnscat2 as "dsncat2" in its extracted
    # PCAP filenames. Keep the source spelling intact on disk, but normalize
    # it to the project's canonical class label in the manifest.
    aliases = {
        "sms malware": "smsmalware",
        "dsncat2": "dnscat2",
        **{label: label for label in CLASS_LABELS},
    }
    matches = {label for token, label in aliases.items() if token in normalized}
    return next(iter(matches)) if len(matches) == 1 else None


def infer_capture_provenance(relative: Path) -> dict:
    """Derive auditable grouping metadata from official dataset paths.

    Unknown layouts deliberately return no group instead of pretending that
    filename fragments have a meaning. An optional provenance sidecar may
    still provide/override the same fields for future datasets.
    """
    if relative.suffix.lower() not in CAPTURE_SUFFIXES:
        return {}
    parts = relative.parts
    lower_parts = [part.lower() for part in parts]
    dataset = lower_parts[0] if lower_parts else ""

    if dataset == "cic-andmal2017" and "extracted" in lower_parts:
        extracted = lower_parts.index("extracted")
        tail = parts[extracted + 1 :]
        if len(tail) < 3:
            return {}
        label = tail[0].lower()
        capture_id = relative.stem
        if label == "benign":
            package_id = ANDMAL_BENIGN_PREFIX_RE.sub("", capture_id).strip().lower()
            if not package_id:
                return {}
            return {
                "sample_id": f"android-package:{package_id}",
                "capture_id": capture_id,
                "capture_batch": str(tail[1]),
                "group_id": f"andmal:benign-app:{package_id}",
                "group_basis": "android_package",
            }
        match = ANDMAL_HASH_RE.search(relative.name)
        if not match:
            return {}
        sample_md5 = match.group(1).lower()
        family = str(tail[1])
        return {
            "sample_id": f"md5:{sample_md5}",
            "sample_md5": sample_md5,
            "capture_id": capture_id,
            "family": family,
            # Hold out entire malware families, not merely individual APKs.
            "group_id": f"andmal:family:{label}:{family.lower()}",
            "group_basis": "malware_family",
        }

    if dataset == "cira-cic-dohbrw-2020" and "dohmalicious" in lower_parts:
        match = DOH_MALICIOUS_RE.match(relative.name)
        if not match:
            return {}
        values = {key: value.lower() for key, value in match.groupdict().items()}
        tool = "dnscat2" if values["tool"] == "dsncat2" else values["tool"]
        resolver = DOH_RESOLVERS[values["resolver"]]
        scenario = f"{tool}:{values['configuration']}:{resolver}"
        return {
            "capture_id": relative.stem,
            "capture_batch": values["timestamp"],
            "tool": tool,
            "device": f"doh-client-{values['client']}",
            "resolver": resolver,
            "scenario": scenario,
            # Ten clients and repeated timestamps from the same configured
            # experiment remain together, preventing scenario duplication.
            "group_id": f"dohbrw:malicious-scenario:{scenario}",
            "group_basis": "tool_configuration_resolver",
        }

    if dataset == "cira-cic-dohbrw-2020" and "dohbenign-nondoh" in lower_parts:
        marker = lower_parts.index("dohbenign-nondoh")
        tail = parts[marker + 1 :]
        if len(tail) < 2:
            return {}
        resolver = str(tail[0]).lower()
        browser = "chrome" if relative.name.lower().startswith("dump_") else "firefox"
        scenario = f"{browser}:{resolver}"
        return {
            "capture_id": "/".join(tail),
            "capture_batch": relative.stem,
            "tool": browser,
            "browser": browser,
            "resolver": resolver,
            "scenario": scenario,
            "group_id": f"dohbrw:benign-scenario:{scenario}",
            "group_basis": "browser_resolver",
        }

    return {}


def should_inventory(relative: Path) -> bool:
    """Inventory extracted captures/CSVs, not archives or nested ZIP copies."""
    return "archives" not in {part.lower() for part in relative.parts} and relative.suffix.lower() in INVENTORY_SUFFIXES


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source_dir: Path = DATASET_SOURCE_DIR, provenance: dict[str, dict] | None = None) -> dict:
    provenance = provenance or {}
    files = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source_dir)
            if not should_inventory(relative):
                continue
            digest = sha256(path)
            record = {
                "path": relative.as_posix(),
                "dataset": relative.parts[0] if relative.parts else "",
                "file_type": path.suffix.lower().lstrip("."),
                "label": infer_label(relative),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
            is_capture = record["file_type"] in {"pcap", "pcapng"}
            record["usable"] = not (is_capture and record["size_bytes"] == 0)
            record["exclusion_reason"] = "empty_capture" if not record["usable"] else None
            inferred = infer_capture_provenance(relative)
            supplied = provenance.get(relative.as_posix(), {})
            for field in PROVENANCE_FIELDS:
                record[field] = supplied.get(field, inferred.get(field))
            if supplied:
                record["provenance_status"] = "verified"
            elif inferred:
                record["provenance_status"] = "derived_from_official_path"
            else:
                record["provenance_status"] = "pcap_only"
            files.append(record)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_dir),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATASET_MANIFESTS_DIR / "source_manifest.json")
    parser.add_argument(
        "--provenance",
        type=Path,
        help="可选 JSON：以 source 相对路径为键，补充 sample_id/capture_id/apk_sha256 等来源字段",
    )
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_manifest(provenance=provenance), ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
