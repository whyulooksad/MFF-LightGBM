import pytest

from user_app.inference.model_loader import load_active_bundle


def test_unpublished_bundle_fails_closed_until_retrained():
    with pytest.raises(FileNotFoundError, match="生产模型包缺少文件"):
        load_active_bundle()
