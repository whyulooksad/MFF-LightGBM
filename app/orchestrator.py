# app/orchestrator.py
import asyncio
import uuid
from datetime import datetime, timezone

from tasks_store import TasksStore
from pipeline_runner import PipelineRunner


class Orchestrator:
    def __init__(self, store: TasksStore, runner: PipelineRunner):
        self.store = store
        self.runner = runner
        self._real_lock = asyncio.Lock()
        self._tasks_in_flight: dict[str, asyncio.Task] = {}
        self._queues: dict[str, asyncio.Queue] = {}

    async def submit(self, mode: str, pcap_path: str | None) -> str:
        task_id = f"T-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.store.create_task(task_id, mode, pcap_path)
        self._queues[task_id] = asyncio.Queue()
        if mode in ("real_test", "real_unknown"):
            await self._real_lock.acquire()
        coro = self._run(task_id, mode, pcap_path)
        self._tasks_in_flight[task_id] = asyncio.create_task(coro)
        return task_id

    async def _run(self, task_id, mode, pcap_path):
        async def router(event_name, data):
            await self._queues[task_id].put((event_name, data))
        try:
            if mode == "demo":
                await self.runner.run_demo(task_id, router)
            else:
                await self.runner.run_real(task_id, mode, pcap_path, router)
            self.store.update_task(task_id, status="done",
                                    finished_at=datetime.now(timezone.utc).isoformat())
        except asyncio.CancelledError:
            self.store.update_task(task_id, status="cancelled",
                                    finished_at=datetime.now(timezone.utc).isoformat())
            raise
        except Exception as e:
            self.store.update_task(task_id, status="failed", error_msg=str(e),
                                    finished_at=datetime.now(timezone.utc).isoformat())
            await self._queues[task_id].put(("task_error", {"task_id": task_id, "error": str(e)}))
        finally:
            await self._queues[task_id].put(("_close", {}))
            # 给 stream reader 一点时间消费 _close，再清 queue
            await asyncio.sleep(0.01)
            self._queues.pop(task_id, None)
            self._tasks_in_flight.pop(task_id, None)
            if mode in ("real_test", "real_unknown"):
                self._real_lock.release()

    def get_queue(self, task_id) -> asyncio.Queue | None:
        return self._queues.get(task_id)

    async def cancel(self, task_id) -> bool:
        t = self._tasks_in_flight.get(task_id)
        if t and not t.done():
            t.cancel()
            return True
        return False
