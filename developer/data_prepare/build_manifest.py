"""Build a reproducible inventory with sizes and SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from developer.config import DATASET_MANIFESTS_DIR, DATASET_SOURCE_DIR
from user_app.inference.contract import CLASS_LABELS


PROVENANCE_FIELDS = (
    "group_id", "group_basis", "sample_id", "capture_id", "apk_sha256",
    "family", "device", "capture_batch", "tool", "resolver",
)


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
        if path.is_file() and path.name.lower() != "readme.txt":
            digest = sha256(path)
            relative = path.relative_to(source_dir)
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
            supplied = provenance.get(relative.as_posix(), {})
            for field in PROVENANCE_FIELDS:
                record[field] = supplied.get(field)
            record["provenance_status"] = "verified" if any(record[field] for field in PROVENANCE_FIELDS) else "pcap_only"
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
