# app/server.py
import json
import os
import sys
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from data_provider import DataProvider
from orchestrator import Orchestrator
from pipeline_runner import PipelineRunner
from tasks_store import TasksStore

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
FLOW_FEATURES_DIR = os.path.join(DATA_DIR, "flow_features")
PIPELINE_OUTPUT_DIR = os.path.join(DATA_DIR, "pipeline", "output")
RUNTIME_DIR = os.path.join(DATA_DIR, "web_runtime")
FRONTEND_DIR = os.path.join(BASE, "frontend", "protypes")

# The web app now lives in the repository root alongside the complete experiment
# source. Reuse the repository-level environment instead of the removed delivery
# bundle's separate virtual environment.
if sys.platform == "win32":
    _default_ml_python = os.path.join(BASE, ".venv", "Scripts", "python.exe")
else:
    _default_ml_python = os.path.join(BASE, ".venv", "bin", "python")
ML_PYTHON = os.environ.get("ML_PYTHON", _default_ml_python)

UPLOAD_DIR = os.path.join(RUNTIME_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="加密流量异常检测系统")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = TasksStore(os.path.join(os.path.dirname(__file__), "tasks.db"))
runner = PipelineRunner(ml_python=ML_PYTHON, project_root=BASE, runtime_dir=RUNTIME_DIR)
orch = Orchestrator(store, runner)
provider = DataProvider(
    flow_features_dir=FLOW_FEATURES_DIR,
    pipeline_output_dir=PIPELINE_OUTPUT_DIR,
    runtime_dir=RUNTIME_DIR,
)

# Serve the migrated frontend from the same origin as the API. This makes the
# application available at http://127.0.0.1:8000/ without opening local files.
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/api/analyze")
async def analyze(mode: str = Form(...), file: Optional[UploadFile] = File(None)):
    if mode not in ("demo", "real_test", "real_unknown"):
        raise HTTPException(400, "mode 必须为 demo / real_test / real_unknown")
    pcap_path = None
    if mode in ("real_test", "real_unknown"):
        if mode == "real_unknown":
            raise HTTPException(400, "根目录正式流水线暂不支持无标签 PCAP 推理")
        if not file:
            raise HTTPException(400, "real_test/real_unknown 模式必须上传 pcap")
        filename = os.path.basename(file.filename or "")
        if not filename.lower().endswith((".pcap", ".pcapng")):
            raise HTTPException(400, "仅支持 .pcap / .pcapng 文件")
        pcap_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
        with open(pcap_path, "wb") as f:
            f.write(await file.read())
    task_id = await orch.submit(mode, pcap_path)
    return {"task_id": task_id, "mode": mode}


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


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = await orch.cancel(task_id)
    return {"cancelled": ok}


@app.get("/api/metadata")
async def metadata(limit: int = 50, offset: int = 0):
    return provider.read_metadata(limit, offset)


@app.get("/api/metadata/{flow_uid}")
async def metadata_detail(flow_uid: str):
    d = provider.read_metadata_detail(flow_uid)
    if not d: raise HTTPException(404)
    return d


@app.get("/api/predictions")
async def predictions(source: str = "runtime", limit: int = 50, offset: int = 0, label: Optional[str] = None):
    return provider.read_predictions(source, limit, offset, label)


@app.get("/api/evaluation")
async def evaluation(source: str = "static"):
    return provider.read_evaluation(source)


@app.get("/api/evaluation/image/{name}")
async def eval_image(name: str, source: str = "static"):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid image name")
    img = provider.read_image_bytes(f"{name}.png" if not name.endswith(".png") else name, source)
    if img is None: raise HTTPException(404)
    return Response(content=img, media_type="image/png")


@app.get("/api/dashboard")
async def dashboard():
    return provider.get_dashboard_stats()


@app.get("/api/labels")
async def labels():
    return [
        {"id": 0, "name": "benign"}, {"id": 1, "name": "adware"},
        {"id": 2, "name": "dns2tcp"}, {"id": 3, "name": "dnscat2"},
        {"id": 4, "name": "iodine"}, {"id": 5, "name": "ransomware"},
        {"id": 6, "name": "scareware"}, {"id": 7, "name": "smsmalware"},
    ]
