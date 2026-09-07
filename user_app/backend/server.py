"""FastAPI entry point for user-facing PCAP detection."""
import json
import os
import sys
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from user_app.backend.data_provider import DataProvider
from user_app.backend.orchestrator import Orchestrator
from user_app.backend.pipeline_runner import PipelineRunner
from user_app.backend.tasks_store import TasksStore
from user_app.inference.config import (
    ROOT,
    RUNTIME_DIR,
    RUNTIME_STATE_DIR,
    RUNTIME_UPLOADS_DIR,
    WEB_STATIC_DIR,
)
from user_app.inference.model_loader import load_active_bundle
from user_app.inference.contract import CLASS_LABELS

BASE = str(ROOT)
FRONTEND_DIR = str(WEB_STATIC_DIR)

# The web app now lives in the repository root alongside the complete experiment
# source. Reuse the repository-level environment instead of the removed delivery
# bundle's separate virtual environment.
if sys.platform == "win32":
    _default_ml_python = os.path.join(BASE, ".venv", "Scripts", "python.exe")
else:
    _default_ml_python = os.path.join(BASE, ".venv", "bin", "python")
ML_PYTHON = os.environ.get("ML_PYTHON", _default_ml_python)

UPLOAD_DIR = str(RUNTIME_UPLOADS_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNTIME_STATE_DIR, exist_ok=True)

app = FastAPI(title="加密流量异常检测系统")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = TasksStore(str(RUNTIME_STATE_DIR / "tasks.db"))
runner = PipelineRunner(ml_python=ML_PYTHON, project_root=BASE, runtime_dir=str(RUNTIME_DIR))
orch = Orchestrator(store, runner)
provider = DataProvider(runtime_dir=str(RUNTIME_DIR))

# Serve the migrated frontend from the same origin as the API. This makes the
# application available at http://127.0.0.1:8000/ without opening local files.
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/status")
async def status():
    try:
        load_active_bundle()
        return {"ready": True, "message": "检测服务已就绪"}
    except (FileNotFoundError, ValueError) as exc:
        return {"ready": False, "message": "检测模型正在准备中"}


@app.post("/api/analyze")
async def analyze(file: Optional[UploadFile] = File(None)):
    try:
        load_active_bundle()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, "检测模型尚未就绪，请稍后再试") from exc
    pcap_path = None
    if not file:
        raise HTTPException(400, "必须上传 pcap")
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith((".pcap", ".pcapng")):
        raise HTTPException(400, "仅支持 .pcap / .pcapng 文件")
    pcap_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    with open(pcap_path, "wb") as f:
        f.write(await file.read())
    task_id = await orch.submit(pcap_path, filename)
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}/stream")
async def stream(task_id: str):
    q = orch.get_queue(task_id)
    if not q:
        raise HTTPException(404, "task stream not active")

    async def gen():
        while True:
            event_name, data = await q.get()
            if event_name == "_close":
                break
            yield {"event": event_name, "data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(gen())


@app.get("/api/tasks")
async def list_tasks(limit: int = 20):
    return store.list_tasks(limit)


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = store.get_task(task_id)
    if not t: raise HTTPException(404)
    return t


@app.get("/api/tasks/{task_id}/summary")
async def task_summary(task_id: str):
    if not store.get_task(task_id):
        raise HTTPException(404)
    summary = provider.read_task_summary(task_id)
    if summary is None:
        raise HTTPException(404, "任务尚未产出结果")
    return summary


@app.get("/api/tasks/{task_id}/predictions")
async def task_predictions(
    task_id: str,
    limit: int = 50,
    offset: int = 0,
    label: Optional[str] = None,
):
    if not store.get_task(task_id):
        raise HTTPException(404)
    return provider.read_task_predictions(task_id, limit, offset, label)


@app.get("/api/tasks/{task_id}/flows")
async def task_flows(task_id: str, limit: int = 50, offset: int = 0):
    if not store.get_task(task_id):
        raise HTTPException(404)
    return provider.read_task_flows(task_id, limit, offset)


@app.get("/api/tasks/{task_id}/flows/{flow_uid}")
async def task_flow_detail(task_id: str, flow_uid: str):
    if not store.get_task(task_id):
        raise HTTPException(404)
    detail = provider.read_task_flow_detail(task_id, flow_uid)
    if detail is None:
        raise HTTPException(404)
    return detail


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = await orch.cancel(task_id)
    return {"cancelled": ok}


@app.get("/api/labels")
async def labels():
    return [{"id": index, "name": name} for index, name in enumerate(CLASS_LABELS)]
