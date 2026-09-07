import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock
from user_app.backend.pipeline_runner import PipelineRunner, parse_marker


class FakeStream:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.lines:
            raise StopAsyncIteration
        return self.lines.pop(0)


class FakeProc:
    def __init__(self, lines, return_code=0):
        self.stdout = FakeStream(lines)
        self.return_code = return_code

    async def wait(self):
        return self.return_code


def test_parse_marker_variants():
    assert parse_marker("@@STAGE:2") == ("stage", 2)
    assert parse_marker('@@STAGE_DONE:2:{"duration_sec": 1.5}') == ("stage_done", 2, {"duration_sec": 1.5})
    assert parse_marker("@@PROGRESS:4:0.500") == ("progress", 4, 0.5)
    assert parse_marker('@@TASK_DONE:{"total_flows": 3}') == ("task_done", {"total_flows": 3})
    assert parse_marker("普通日志行") is None
    assert parse_marker("@@STAGE:99") is None       # 超出阶段范围
    assert parse_marker("@@STAGE:abc") is None      # 非法数字
    assert parse_marker("@@TASK_DONE:not-json") is None
    assert parse_marker('@@STAGE_DONE:1:') == ("stage_done", 1, {})


@pytest.mark.asyncio
async def test_run_real_uses_inference_without_labels(tmp_path):
    pcap = tmp_path / "my_capture.pcap"
    pcap.write_bytes(b"fake")

    runner = PipelineRunner(
        ml_python="/path/to/python",
        project_root=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    events = []
    async def cb(evt, data):
        events.append((evt, data))

    lines = [
        "some log (1/10)",
        "@@STAGE:1",
        "@@STAGE_DONE:1:{\"duration_sec\": 0.1}",
        "@@STAGE:2",
        "@@PROGRESS:2:0.4",
        "@@STAGE_DONE:2:{\"duration_sec\": 0.2}",
        "@@STAGE:3", "@@STAGE_DONE:3:{}",
        "@@STAGE:4", "@@STAGE_DONE:4:{}",
        "@@STAGE:5", "@@STAGE_DONE:5:{}",
        '@@TASK_DONE:{"total_flows": 5, "elapsed_sec": 3.2}',
    ]
    fake_proc = FakeProc(lines)

    async def fake_exec(*args, **kwargs):
        # 断言调用的是纯推理入口，且不依赖文件名标签
        assert "-m" in args and "user_app.inference.pipeline" in args
        assert "--pcap" in args and "--output-dir" in args
        assert "main.py" not in args
        return fake_proc

    with patch("user_app.backend.pipeline_runner.asyncio.create_subprocess_exec", new=fake_exec):
        await runner.run_real("T-1", str(pcap), cb)

    stage_starts = [d["stage"] for e, d in events if e == "stage_start"]
    assert stage_starts == [1, 2, 3, 4, 5]
    done_events = [d for e, d in events if e == "task_done"]
    assert len(done_events) == 1
    assert done_events[0]["redirect"] == "detection.html?task=T-1"
    assert done_events[0]["summary"]["total_flows"] == 5


@pytest.mark.asyncio
async def test_run_real_rejects_non_pcap(tmp_path):
    runner = PipelineRunner(
        ml_python="/path/to/python",
        project_root=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    bad = tmp_path / "capture.txt"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        await runner.run_real("T-2", str(bad), cb=None)
