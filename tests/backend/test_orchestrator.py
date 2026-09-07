# app/tests/test_orchestrator.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from user_app.backend.orchestrator import Orchestrator

@pytest.mark.asyncio
async def test_submit_real_creates_queue_and_task(tmp_path):
    store = MagicMock()
    runner = MagicMock()
    runner.run_real = AsyncMock()
    runner.task_dir.return_value = tmp_path / "runtime" / "tasks" / "T-test"
    orch = Orchestrator(store, runner)
    task_id = await orch.submit(str(tmp_path / "capture.pcap"), "capture.pcap")
    assert task_id.startswith("T-")
    assert orch.get_queue(task_id) is not None
    store.create_task.assert_called_once()
    # 等任务跑完
    await asyncio.sleep(0.05)
    assert orch.get_queue(task_id) is None  # 任务结束后 queue 被清理

@pytest.mark.asyncio
async def test_cancel_running_task():
    store = MagicMock()
    runner = MagicMock()
    async def slow_real(tid, pcap, cb):
        await asyncio.sleep(10)
    runner.run_real = slow_real
    runner.task_dir.return_value = "/tmp/runtime/T-test"
    orch = Orchestrator(store, runner)
    tid = await orch.submit("/tmp/capture.pcap")
    ok = await orch.cancel(tid)
    assert ok is True
