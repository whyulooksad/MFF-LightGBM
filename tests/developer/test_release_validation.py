import json

from developer.release.validate_bundle import validate_bundle
from user_app.inference.contract import CLASS_LABELS, FEATURE_SCHEMA_VERSION, NEW_FORMAT_NUM_FEATURES


def test_release_validation_rejects_empty_encoder_and_lora_directories(tmp_path):
    (tmp_path / "encoder").mkdir()
    (tmp_path / "lora").mkdir()
    (tmp_path / "supcon").mkdir()
    (tmp_path / "detector").mkdir()
    (tmp_path / "supcon" / "reducer.pt").write_bytes(b"placeholder")
    (tmp_path / "detector" / "best_lgb_model.pkl").write_bytes(b"placeholder")
    columns = [f"feat_{index}" for index in range(64)] + list(NEW_FORMAT_NUM_FEATURES)
    (tmp_path / "detector" / "feature_columns.json").write_text(json.dumps(columns), encoding="utf-8")
    (tmp_path / "detector" / "imputation_medians.json").write_text(
        json.dumps({column: 0.0 for column in columns}), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION}), encoding="utf-8")
    (tmp_path / "feature_schema.json").write_text(json.dumps({
        "schema_version": FEATURE_SCHEMA_VERSION, "numeric_features": NEW_FORMAT_NUM_FEATURES,
    }), encoding="utf-8")
    (tmp_path / "label_mapping.json").write_text(json.dumps({"labels": CLASS_LABELS}), encoding="utf-8")

    errors = validate_bundle(tmp_path)
    assert "encoder 中没有包含配置和权重的 checkpoint-*" in errors
    assert "lora/best 中缺少 adapter 配置或权重" in errors
