"""Run one uploaded PCAP through pure production inference."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from pathlib import Path


PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
MARKER_RE = re.compile(r"^@@(STAGE_DONE|STAGE|PROGRESS|TASK_DONE):(.*)$")


def parse_marker(line: str):
    """Parse one ``user_app.inference.pipeline`` stdout marker line.

    Returns a tuple event or ``None`` when the line is a plain log line.
    """
    match = MARKER_RE.match(line)
    if not match:
        return None
    kind, rest = match.group(1), match.group(2)
    try:
        if kind == "STAGE":
            stage_no = int(rest)
            if not 1 <= stage_no <= PipelineRunner.NUM_STAGES:
                return None
            return ("stage", stage_no)
        if kind == "STAGE_DONE":
            stage_part, _, payload = rest.partition(":")
            stage_no = int(stage_part)
            stats = json.loads(payload) if payload else {}
            return ("stage_done", stage_no, stats)
        if kind == "PROGRESS":
            stage_part, _, fraction = rest.partition(":")
            return ("progress", int(stage_part), float(fraction))
        if kind == "TASK_DONE":
            return ("task_done", json.loads(rest))
    except (ValueError, json.JSONDecodeError):
        return None
    return None


class PipelineRunner:
    STAGE_NAMES = [
        "读取抓包文件",
        "整理网络连接",
        "分析连接行为",
        "识别安全风险",
        "生成检测结果",
    ]
    NUM_STAGES = len(STAGE_NAMES)

    def __init__(self, ml_python: str, project_root: str, runtime_dir: str):
        self.ml_python = ml_python
        self.project_root = Path(project_root).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.upload_dir = self.runtime_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        return self.runtime_dir / "tasks" / task_id

    @staticmethod
    def _write_task_readme(task_dir: Path) -> None:
        (task_dir / "readme.txt").write_text(
            "本目录属于一次用户 PCAP 检测任务。\n"
            "包含上传副本、流特征、逐流预测和 summary.json。\n"
            "这些文件不是训练数据，也不会用于更新生产模型。\n",
            encoding="utf-8",
        )

    async def run_real(self, task_id, pcap_path, cb, original_filename=None):
        """Pure inference on one uploaded PCAP; frozen checkpoints, no training."""
        if not pcap_path:
            raise ValueError("未提供 PCAP 文件")
        source = Path(pcap_path)
        lower_name = source.name.lower()
        if not lower_name.endswith((".pcap", ".pcapng")):
            raise ValueError("仅支持 .pcap / .pcapng 文件")

        task_dir = self.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        self._write_task_readme(task_dir)
        target = task_dir / f"input{source.suffix.lower()}"
        shutil.move(str(source), str(target))

        await cb("task_start", {
            "task_id": task_id,
            "stages": self.NUM_STAGES,
            "total_estimate_sec": 60,
        })

        proc = await asyncio.create_subprocess_exec(
            self.ml_python,
            "-m", "user_app.inference.pipeline",
            "--pcap", str(target),
            "--output-dir", str(task_dir),
            cwd=str(self.project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None

        started = time.monotonic()
        current_stage = None
        summary = None

        async for raw in proc.stdout:
            text = raw.decode("utf-8", errors="replace").rstrip()
            event = parse_marker(text)
            if event is not None:
                kind = event[0]
                if kind == "stage":
                    current_stage = event[1]
                    await cb("stage_start", {
                        "task_id": task_id,
                        "stage": current_stage,
                        "name": self.STAGE_NAMES[current_stage - 1],
                    })
                elif kind == "stage_done":
                    await cb("stage_done", {
                        "task_id": task_id,
                        "stage": event[1],
                        "duration_sec": event[2].get("duration_sec", 0),
                        "stats": event[2],
                    })
                elif kind == "progress":
                    await cb("stage_progress", {
                        "task_id": task_id,
                        "stage": event[1],
                        "progress": event[2],
                    })
                elif kind == "task_done":
                    summary = event[1]
                continue

            progress = None
            if current_stage is not None:
                match = PROGRESS_RE.search(text)
                if match:
                    current, total = int(match.group(1)), int(match.group(2))
                    progress = round(current / total, 3) if total else None
            await cb("stage_progress", {
                "task_id": task_id,
                "stage": current_stage,
                "progress": progress,
            })

        return_code = await proc.wait()
        if return_code != 0:
            raise RuntimeError(f"推理进程失败: exit {return_code}")
        if summary is None:
            raise RuntimeError("推理进程未输出结果摘要 (missing @@TASK_DONE)")
        if original_filename:
            summary["pcap_filename"] = original_filename
            summary_path = task_dir / "summary.json"
            if summary_path.exists():
                stored = json.loads(summary_path.read_text(encoding="utf-8"))
                stored["pcap_filename"] = original_filename
                summary_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")

        await cb("task_done", {
            "task_id": task_id,
            "total_duration_sec": summary.get("elapsed_sec") or round(time.monotonic() - started, 1),
            "redirect": f"detection.html?task={task_id}",
            "summary": summary,
        })
