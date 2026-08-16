# app/tests/test_pipeline_runner.py
import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from pipeline_runner import PipelineRunner

@pytest.mark.asyncio
async def test_demo_uses_canonical_outputs(tmp_path):
    output = tmp_path / "data" / "pipeline" / "output"
    features = tmp_path / "data" / "flow_features"
    output.mkdir(parents=True)
    features.mkdir(parents=True)
    for path in (
        features / "flow_metadata_temporal_test.csv",
        features / "final_multiclass_features_test.csv",
        output / "predictions.csv",
        output / "classification_report.txt",
    ):
        path.write_text("test", encoding="utf-8")
    runner = PipelineRunner(
        ml_python="/path/to/python",
        project_root=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    events = []
    async def cb(evt, data):
        events.append((evt, data))
    with patch("pipeline_runner.asyncio.create_subprocess_exec", new=AsyncMock()) as mock_exec:
        await runner.run_demo("T-001", cb)
    # 至少 stage_start/stage_done 各 5 次 + task_done 1 次
    assert any(e[0] == "task_done" for e in events)
    assert mock_exec.call_count == 0
