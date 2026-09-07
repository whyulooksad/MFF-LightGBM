import pytest

from user_app.inference import model_loader
from user_app.inference.model_loader import ProductionBundle, load_active_bundle


def test_unpublished_bundle_fails_closed_until_retrained():
    with pytest.raises(FileNotFoundError, match="生产模型包缺少文件"):
        load_active_bundle()


def test_active_bundle_is_resolved_each_time_for_hot_release_switch(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    production = tmp_path / "production"
    monkeypatch.setattr(model_loader, "ACTIVE_MODEL_FILE", active)
    monkeypatch.setattr(model_loader, "PRODUCTION_ROOT", production)

    active.write_text('{"release_id":"release-a"}', encoding="utf-8")
    first = ProductionBundle.active()
    active.write_text('{"release_id":"release-b"}', encoding="utf-8")
    second = ProductionBundle.active()

    assert first.root == production / "release-a"
    assert second.root == production / "release-b"
