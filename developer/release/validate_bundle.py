"""Validate a candidate model bundle before publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from user_app.inference.contract import CLASS_LABELS, FEATURE_SCHEMA_VERSION, NEW_FORMAT_NUM_FEATURES


REQUIRED = (
    "encoder", "lora", "supcon/reducer.pt",
    "detector/best_lgb_model.pkl", "detector/feature_columns.json", "detector/imputation_medians.json",
    "manifest.json", "feature_schema.json", "label_mapping.json",
)


def validate_bundle(root: Path) -> list[str]:
    errors = [f"缺少 {name}" for name in REQUIRED if not (root / name).exists()]
    labels_path = root / "label_mapping.json"
    if labels_path.exists():
        try:
            if json.loads(labels_path.read_text(encoding="utf-8")).get("labels") != CLASS_LABELS:
                errors.append("label_mapping.json 的标签或顺序与 contract.py 不一致")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"label_mapping.json 无法读取: {exc}")
    schema_path, manifest_path = root / "feature_schema.json", root / "manifest.json"
    if schema_path.exists() and manifest_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            schema_version = str(schema.get("schema_version", ""))
            if schema_version != FEATURE_SCHEMA_VERSION:
                errors.append(f"feature_schema.json 必须使用当前特征契约 {FEATURE_SCHEMA_VERSION}，实际为 {schema_version or '空'}")
            if str(manifest.get("feature_schema_version", "")) != schema_version:
                errors.append("manifest.json 与 feature_schema.json 的特征契约不一致")
            declared_features = schema.get("numeric_features")
            if declared_features != NEW_FORMAT_NUM_FEATURES:
                errors.append("feature_schema.json 的 80 维数值特征或顺序与 contract.py 不一致")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"特征契约无法读取: {exc}")
    feature_columns_path = root / "detector" / "feature_columns.json"
    if feature_columns_path.exists():
        try:
            columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
            allowed = set(NEW_FORMAT_NUM_FEATURES) | {f"feat_{index}" for index in range(64)}
            unknown = [column for column in columns if column not in allowed]
            missing_semantic = [f"feat_{index}" for index in range(64) if f"feat_{index}" not in columns]
            if len(columns) != len(set(columns)):
                errors.append("detector/feature_columns.json 包含重复列")
            if unknown:
                errors.append(f"detector 包含契约外特征: {unknown[:8]}")
            if missing_semantic:
                errors.append(f"detector 缺少 SupCon-AE 输出列: {missing_semantic[:8]}")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"detector/feature_columns.json 无法读取: {exc}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    problems = validate_bundle(args.bundle)
    if problems:
        raise SystemExit("模型包校验失败:\n- " + "\n- ".join(problems))
    print("模型包校验通过")
