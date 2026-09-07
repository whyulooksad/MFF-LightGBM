"""Resolve and validate the production model bundle selected by active.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from user_app.inference.config import ACTIVE_RELEASE_ID, PRODUCTION_MODEL_DIR
from user_app.inference.contract import CLASS_LABELS, FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class ProductionBundle:
    release_id: str
    root: Path

    @classmethod
    def active(cls) -> "ProductionBundle":
        return cls(ACTIVE_RELEASE_ID, PRODUCTION_MODEL_DIR)

    @property
    def encoder(self) -> Path:
        return self.root / "encoder"

    @property
    def lora(self) -> Path:
        return self.root / "lora"

    @property
    def supcon(self) -> Path:
        return self.root / "supcon" / "reducer.pt"

    @property
    def detector(self) -> Path:
        return self.root / "detector"

    def required_paths(self) -> list[Path]:
        return [
            self.encoder,
            self.lora,
            self.supcon,
            self.detector / "best_lgb_model.pkl",
            self.detector / "feature_columns.json",
            self.detector / "imputation_medians.json",
            self.root / "manifest.json",
            self.root / "feature_schema.json",
            self.root / "label_mapping.json",
        ]

    def validate(self) -> None:
        missing = [str(path) for path in self.required_paths() if not path.exists()]
        if missing:
            raise FileNotFoundError("生产模型包缺少文件:\n  " + "\n  ".join(missing))
        labels = json.loads((self.root / "label_mapping.json").read_text(encoding="utf-8"))["labels"]
        if labels != CLASS_LABELS:
            raise ValueError("生产模型标签顺序与运行时 contract.py 不一致")
        schema = json.loads((self.root / "feature_schema.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        packaged = str(schema.get("schema_version", ""))
        declared = str(manifest.get("feature_schema_version", ""))
        if packaged != declared:
            raise ValueError(f"模型包内部特征契约不一致: schema={packaged}, manifest={declared}")
        if packaged != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"模型需要特征契约 {packaged or 'unknown'}，当前解析器为 {FEATURE_SCHEMA_VERSION}；"
                "请用当前开发流程重新训练并发布模型"
            )


def load_active_bundle() -> ProductionBundle:
    bundle = ProductionBundle.active()
    bundle.validate()
    return bundle
