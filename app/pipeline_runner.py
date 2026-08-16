"""Run the repository's canonical pipeline for the Web application."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path


PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


class PipelineRunner:
    STAGE_NAMES = ["PCAP 截断", "人工特征提取", "日志序列化", "DeBERTa+融合", "SupCon-AE + LightGBM 检测"]
    STAGE_KEYS = ["truncate", "flow_features", "preprocess", "extract", "detector"]

    def __init__(self, ml_python: str, project_root: str, runtime_dir: str):
        self.ml_python = ml_python
        self.project_root = Path(project_root).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.upload_dir = self.runtime_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def _subprocess(self, args, cb, task_id, stage_no):
        proc = await asyncio.create_subprocess_exec(
            self.ml_python,
            *args,
            cwd=str(self.project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            match = PROGRESS_RE.search(text)
            progress = None
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                progress = round(current / total, 3) if total else 0
            await cb("stage_progress", {
                "task_id": task_id,
                "stage": stage_no,
                "progress": progress,
                "log": text,
            })
        return await proc.wait()

    async def _emit_simulated(self, cb, task_id, stage_no, name, stats_log):
        await cb("stage_start", {
            "task_id": task_id,
            "stage": stage_no,
            "name": name,
            "script": self.STAGE_KEYS[stage_no - 1],
            "simulated": True,
        })
        for step in range(1, 6):
            await asyncio.sleep(0.04)
            await cb("stage_progress", {
                "task_id": task_id,
                "stage": stage_no,
                "progress": step / 5,
                "log": f"{stats_log} ({step}/5)",
            })
        await cb("stage_done", {
            "task_id": task_id,
            "stage": stage_no,
            "duration_sec": 0.2,
            "stats": {},
        })

    def _require_demo_outputs(self):
        required = [
            self.project_root / "data" / "flow_features" / "flow_metadata_temporal_test.csv",
            self.project_root / "data" / "flow_features" / "final_multiclass_features_test.csv",
            self.project_root / "data" / "pipeline" / "output" / "predictions.csv",
            self.project_root / "data" / "pipeline" / "output" / "classification_report.txt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("演示所需的根目录实验产物缺失: " + ", ".join(missing))

    async def run_demo(self, task_id, cb):
        """Replay progress while serving the canonical experiment outputs."""
        self._require_demo_outputs()
        await cb("task_start", {"task_id": task_id, "mode": "demo", "stages": 5, "total_estimate_sec": 1})
        logs = [
            "已使用根目录 PCAP 截断产物",
            "已加载流统计与 TLS/X509 特征",
            "已加载流日志 JSONL",
            "已加载 DeBERTa/LoRA 融合特征",
            "已加载 SupCon-AE 降维与 LightGBM 八分类检测结果",
        ]
        for index, (name, log) in enumerate(zip(self.STAGE_NAMES, logs), start=1):
            await self._emit_simulated(cb, task_id, index, name, log)
        await cb("task_done", {"task_id": task_id, "total_duration_sec": 1, "redirect": "detection.html"})

    async def run_real(self, task_id, mode, pcap_path, cb):
        """Run the canonical root pipeline against an uploaded labelled test PCAP.

        The current canonical experiment pipeline derives the class label from
        ``<label>_..._test.pcap``. Unknown/unlabelled inference is deliberately
        rejected instead of silently assigning a false ground-truth label.
        """
        if mode == "real_unknown":
            raise ValueError("根目录正式流水线当前需要标签文件名，暂不支持 real_unknown")
        if not pcap_path:
            raise ValueError("未提供 PCAP 文件")

        source = Path(pcap_path)
        lower_name = source.name.lower()
        labels = ("benign", "adware", "dns2tcp", "dnscat2", "iodine", "ransomware", "scareware", "smsmalware")
        label = next((item for item in labels if lower_name.startswith(item + "_")), None)
        if label is None:
            raise ValueError("real_test 文件名必须以八类标签之一开头")

        raw_dir = self.project_root / "data" / "pcap" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pcapng" if lower_name.endswith(".pcapng") else ".pcap"
        target = raw_dir / f"{label}_sampled_web_{task_id}_test{suffix}"
        shutil.copy2(source, target)

        await cb("task_start", {"task_id": task_id, "mode": mode, "stages": 5, "total_estimate_sec": 900})
        for index, (stage, name) in enumerate(zip(self.STAGE_KEYS, self.STAGE_NAMES), start=1):
            await cb("stage_start", {
                "task_id": task_id,
                "stage": index,
                "name": name,
                "script": f"main.py --stage {stage}",
                "simulated": False,
            })
            rc = await self._subprocess(["main.py", "--stage", stage], cb, task_id, index)
            if rc != 0:
                raise RuntimeError(f"正式流水线阶段 {stage} 执行失败: exit {rc}")
            await cb("stage_done", {"task_id": task_id, "stage": index, "duration_sec": 0, "stats": {}})
        await cb("task_done", {"task_id": task_id, "total_duration_sec": 900, "redirect": "detection.html"})
