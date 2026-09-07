"""Copy a validated experiment bundle into a new immutable production release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from developer.release.validate_bundle import validate_bundle
from user_app.inference.config import ACTIVE_MODEL_FILE, PRODUCTION_ROOT


def publish(source: Path, release_id: str, activate: bool = False) -> Path:
    if not release_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in release_id):
        raise ValueError("发布标识只能包含字母、数字、点、下划线和连字符")
    errors = validate_bundle(source)
    if errors:
        raise ValueError("候选模型未通过校验:\n- " + "\n- ".join(errors))
    destination = PRODUCTION_ROOT / release_id
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("release_id")) != release_id:
        raise ValueError(f"manifest.json release_id 必须等于发布标识 {release_id}")
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已发布版本: {destination}")
    shutil.copytree(source, destination)
    published_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    published_manifest["status"] = "production"
    (destination / "manifest.json").write_text(
        json.dumps(published_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if activate:
        ACTIVE_MODEL_FILE.write_text(json.dumps({"release_id": release_id}, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("release_id")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    print(publish(args.source, args.release_id, args.activate))
