# app/tests/test_orchestrator.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from orchestrator import Orchestrator

@pytest.mark.asyncio
async def test_submit_demo_creates_queue_and_task(tmp_path):
    store = MagicMock()
    runner = AsyncMock()
    runner.run_demo = AsyncMock()
    orch = Orchestrator(store, runner)
    task_id = await orch.submit("demo", None)
    assert task_id.startswith("T-")
    assert orch.get_queue(task_id) is not None
    store.create_task.assert_called_once()
    # 等任务跑完
    await asyncio.sleep(0.05)
    assert orch.get_queue(task_id) is None  # 任务结束后 queue 被清理

@pytest.mark.asyncio
async def test_cancel_running_task():
    store = MagicMock()
    runner = AsyncMock()
    async def slow_demo(tid, cb):
        await asyncio.sleep(10)
    runner.run_demo = slow_demo
    orch = Orchestrator(store, runner)
    tid = await orch.submit("demo", None)
    ok = await orch.cancel(tid)
    assert ok is True
