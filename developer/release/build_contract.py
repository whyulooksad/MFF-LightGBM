"""Write the runtime contract files for a completed candidate experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from user_app.inference.contract import CLASS_LABELS, FEATURE_SCHEMA_VERSION, NEW_FORMAT_NUM_FEATURES


def write_contract(root: Path, release_id: str, force: bool = False) -> None:
    if not release_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in release_id):
        raise ValueError("发布标识只能包含字母、数字、点、下划线和连字符")
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / "feature_schema.json", root / "label_mapping.json", root / "manifest.json"]
    if not force and any(path.exists() for path in paths):
        raise FileExistsError("契约文件已存在；确认需要重建时使用 --force")
    schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "numeric_features": NEW_FORMAT_NUM_FEATURES,
        "semantic_dimension": 768,
        "reduced_semantic_dimension": 64,
        "byte_semantics": "IP packet length",
        "missing_value_semantics": "NaN means not observed or not computable; zero is measured zero",
    }
    labels = {"labels": CLASS_LABELS}
    manifest = {
        "release_id": release_id,
        "status": "candidate",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "labels": CLASS_LABELS,
    }
    for path, data in zip(paths, (schema, labels, manifest)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("release_id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_contract(args.experiment, args.release_id, args.force)
