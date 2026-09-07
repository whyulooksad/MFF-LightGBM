"""User-runtime paths and the active immutable production model."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEB_STATIC_DIR = ROOT / "user_app" / "frontend"

DATA_DIR = ROOT / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
RUNTIME_UPLOADS_DIR = RUNTIME_DIR / "uploads"
RUNTIME_TASKS_DIR = RUNTIME_DIR / "tasks"
RUNTIME_STATE_DIR = RUNTIME_DIR / "state"

MODEL_ARTIFACTS_DIR = ROOT / "models"
PRODUCTION_ROOT = MODEL_ARTIFACTS_DIR / "production"
ACTIVE_MODEL_FILE = PRODUCTION_ROOT / "active.json"


def _active_release_id() -> str:
    try:
        value = json.loads(ACTIVE_MODEL_FILE.read_text(encoding="utf-8"))
        release_id = str(value.get("release_id", "")).strip()
        return release_id or "unpublished"
    except (OSError, ValueError, TypeError):
        return "unpublished"


ACTIVE_RELEASE_ID = _active_release_id()
PRODUCTION_MODEL_DIR = PRODUCTION_ROOT / ACTIVE_RELEASE_ID
PRODUCTION_PRETRAIN_DIR = PRODUCTION_MODEL_DIR / "encoder"
PRODUCTION_LORA_DIR = PRODUCTION_MODEL_DIR / "lora"
PRODUCTION_SUPCON_MODEL_PATH = PRODUCTION_MODEL_DIR / "supcon" / "reducer.pt"
PRODUCTION_DETECTOR_DIR = PRODUCTION_MODEL_DIR / "detector"
