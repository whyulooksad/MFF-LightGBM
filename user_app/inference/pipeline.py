# -*- coding: utf-8 -*-
"""Pure inference stage: one PCAP -> per-flow eight-class predictions.

Unlike the offline pipeline (``python -m developer.pipeline --stage all``), this module never
trains anything. It loads the frozen checkpoints produced offline

    models/production/<release_id>/encoder/   RTD encoder
    models/production/<release_id>/lora/      LoRA adapter
    models/production/<release_id>/supcon/    SupCon-AE reducer
    models/production/<release_id>/detector/  LightGBM and schema

and runs: flow reconstruction/features -> log serialization -> DeBERTa [CLS]
-> SupCon-AE 64-d -> + 80 manual features -> LightGBM predict.

Stdout protocol consumed by the web runner (one marker per line):
    @@STAGE:<no>                     stage started
    @@STAGE_DONE:<no>:<json>         stage finished
    @@PROGRESS:<no>:<frac>           progress within a stage (0..1)
    @@TASK_DONE:<json>               final summary
Everything else is human-readable logging.

这份文件做的事情，可以完整翻译成：
用户给我一份 PCAP 和一个结果保存目录。我先确认 PCAP 文件存在，再读取当前激活的生产模型版本，检查 DeBERTa、LoRA、SupCon-AE、LightGBM、标签映射和特征契约是否齐全且互相兼容。
接着我启动一个独立子进程解析 PCAP。子进程把数据包组合成双向流，进行 TCP 重组、TLS 和证书解析，并提取 80 个人工网络特征。如果十分钟仍未完成，我就终止它。
特征提取完成后，我把连接日志、TLS 日志和证书日志整理成 DeBERTa 能读的结构化文本，并保存为 flows.jsonl。
然后我加载生产版 DeBERTa Encoder 和 LoRA Adapter，把流量文本分词、补齐或截断到 512 个 token，按批送入模型，为每条流生成 768 个语义数字。
接着我把这 768 个语义数字与 80 个人工特征组合起来。SupCon-AE 把 768 个语义数字压缩成 64 个，再将 64 个语义特征与 80 个人工特征交给 LightGBM。
LightGBM 为每条流输出 8 个类别概率。我选择概率最大的类别作为最终预测，并把这个最大概率作为置信度。
最后，我通过 flow_uid 把预测结果与源 IP、目标 IP、端口和协议重新合并，写出逐流预测文件 predictions.csv，再统计总流数、恶意流数、恶意比例、类别数量和阶段耗时，写出 summary.json，最后通知网页任务已经完成。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pandas as pd

from user_app.inference.contract import ID2LABEL, NEW_FORMAT_NUM_FEATURES, NUM_LABELS
from user_app.inference.config import (
    PRODUCTION_DETECTOR_DIR,
    PRODUCTION_LORA_DIR,
    PRODUCTION_MODEL_DIR,
    PRODUCTION_PRETRAIN_DIR,
    PRODUCTION_SUPCON_MODEL_PATH,
    ACTIVE_RELEASE_ID,
)
from user_app.inference.extract_flow_features import (
    CSV_HEADERS,
    TEMPORAL_HEADERS,
    process_one_file,
)
from user_app.inference.supcon_reducer import SupConAEReducer, replace_semantic_features
from user_app.inference.serialize_flow import preprocess
from user_app.inference.semantic_encoder import (
    FEATURE_DIM,
    FeatureDataset,
    build_model,
    default_lora_adapter,
    extract_cls_features,
    latest_pretrain_checkpoint,
    load_tokenizer,
)
from user_app.inference.flow_io import load_flows
from user_app.inference.lightgbm_detector import LightGBMDetector
from user_app.inference.model_loader import load_active_bundle


DETECTOR_DIR = PRODUCTION_DETECTOR_DIR
LGB_MODEL_PKL = DETECTOR_DIR / "best_lgb_model.pkl"
FEATURE_COLUMNS_JSON = DETECTOR_DIR / "feature_columns.json"
SUPCON_MODEL_PATH = PRODUCTION_SUPCON_MODEL_PATH

EXTRACT_TIMEOUT_SEC = 600
PREDICT_BATCH_SIZE = 4096

TUPLE_COLUMNS = ["flow_uid", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]


def emit_stage(stage_no: int) -> None:
    print(f"@@STAGE:{stage_no}", flush=True)


def emit_stage_done(stage_no: int, payload: dict | None = None) -> None:
    print(f"@@STAGE_DONE:{stage_no}:{json.dumps(payload or {}, ensure_ascii=False)}", flush=True)


def emit_progress(stage_no: int, fraction: float) -> None:
    fraction = max(0.0, min(1.0, float(fraction)))
    print(f"@@PROGRESS:{stage_no}:{fraction:.3f}", flush=True)


def emit_task_done(summary: dict) -> None:
    print(f"@@TASK_DONE:{json.dumps(summary, ensure_ascii=False)}", flush=True)


def required_checkpoints() -> list[Path]:
    required = [
        LGB_MODEL_PKL,
        FEATURE_COLUMNS_JSON,
        SUPCON_MODEL_PATH,
    ]
    if latest_pretrain_checkpoint(PRODUCTION_PRETRAIN_DIR) is None:
        required.append(PRODUCTION_PRETRAIN_DIR)
    if default_lora_adapter(PRODUCTION_LORA_DIR) is None:
        required.append(PRODUCTION_LORA_DIR / "best")
    return required


def stage_flow_features(pcap_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Run the single-file extractor in a child process, then merge its CSVs."""
    queue: Queue = Queue()
    worker = Process(
        target=process_one_file,
        args=(str(pcap_path), "", "inference", "test", queue),
    )
    worker.start()
    worker.join(timeout=EXTRACT_TIMEOUT_SEC)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        raise RuntimeError(f"特征提取超时（>{EXTRACT_TIMEOUT_SEC}s），已中止")

    try:
        message = queue.get(timeout=10)
    except Exception as exc:  # noqa: BLE001 - child died without reporting
        raise RuntimeError(f"特征提取子进程未返回结果: {exc}") from exc

    if message[0] != "success":
        raise RuntimeError(f"特征提取失败:\n{message[1]}")

    _, ml_tmp, temporal_tmp, ml_count, temporal_count = message
    ml_csv = output_dir / "flow_features.csv"
    temporal_csv = output_dir / "flow_temporal.csv"
    _merge_tmp_csv(ml_tmp, ml_csv, CSV_HEADERS)
    _merge_tmp_csv(temporal_tmp, temporal_csv, TEMPORAL_HEADERS)
    print(f"  流特征: ML {ml_count} 条, 时序 {temporal_count} 条")
    return ml_csv, temporal_csv


def _merge_tmp_csv(tmp_path: str, final_path: Path, columns: list[str]) -> None:
    tmp = Path(tmp_path)
    if tmp.exists():
        frame = pd.read_csv(tmp, dtype=str, keep_default_na=False)
        tmp.unlink()
    else:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(final_path, index=False, encoding="utf-8")


def stage_serialize(features_csv: Path, output_dir: Path) -> Path:
    jsonl_path = output_dir / "flows.jsonl"
    preprocess(
        csv_path=str(features_csv),
        output_path=str(jsonl_path),
        target="feature",
    )
    return jsonl_path


def stage_semantic_features(jsonl_path: Path, batch_size: int) -> tuple[list[dict], list[list[float]]]:
    import torch
    from torch.utils.data import DataLoader

    flows = load_flows(str(jsonl_path))
    if not flows:
        raise RuntimeError("未从 PCAP 中提取到任何双向流，无法推理")

    base_checkpoint = latest_pretrain_checkpoint(PRODUCTION_PRETRAIN_DIR)
    adapter = default_lora_adapter(PRODUCTION_LORA_DIR)
    if base_checkpoint is None or adapter is None:
        raise FileNotFoundError(f"生产模型不完整: {PRODUCTION_MODEL_DIR}")
    model, tokenizer_dir = build_model(
        base_model_dir=base_checkpoint,
        adapter_dir=adapter,
    )
    tokenizer = load_tokenizer(tokenizer_dir)
    dataset = FeatureDataset(flows, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}, 流数: {len(flows):,}, batch={batch_size}")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    emit_progress(4, 0.05)
    features, _ = extract_cls_features(model, dataloader, device)
    emit_progress(4, 1.0)
    if len(features) != len(flows):
        raise RuntimeError("语义特征数量与流数量不一致")
    return flows, features


def stage_predict(flows: list[dict], features: list[list[float]]) -> pd.DataFrame:
    num_cols = list(NEW_FORMAT_NUM_FEATURES)
    records = []
    for flow, vector in zip(flows, features):
        row: dict = {"flow_uid": flow.get("flow_uid", "")}
        for index, value in enumerate(vector):
            row[f"feat_{index}"] = value
        num_features = flow.get("num_features") or {}
        for col in num_cols:
            row[col] = num_features.get(col)
        records.append(row)

    columns = ["flow_uid"] + [f"feat_{index}" for index in range(FEATURE_DIM)] + num_cols
    fused = pd.DataFrame(records, columns=columns)

    emit_progress(5, 0.1)
    reducer = SupConAEReducer.load(SUPCON_MODEL_PATH, verbose=False)
    reduced = replace_semantic_features(fused, reducer)
    emit_progress(5, 0.4)

    detector = LightGBMDetector(DETECTOR_DIR, batch_size=PREDICT_BATCH_SIZE)
    proba = detector.predict_proba(reduced)
    emit_progress(5, 0.9)

    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    result = pd.DataFrame(
        {
            "flow_uid": reduced["flow_uid"].astype(str),
            "pred_label": pred.astype(int),
            "confidence": np.round(confidence, 4),
        }
    )
    for index in range(NUM_LABELS):
        result[f"proba_{ID2LABEL[index]}"] = np.round(proba[:, index], 4)
    result["pred_label_name"] = result["pred_label"].map(ID2LABEL)
    emit_progress(5, 1.0)
    return result


def stage_write_outputs(
    result: pd.DataFrame,
    features_csv: Path,
    output_dir: Path,
    pcap_name: str,
    timings: dict,
) -> dict:
    tuple_df = pd.read_csv(
        features_csv,
        usecols=lambda col: col in TUPLE_COLUMNS,
        dtype=str,
        keep_default_na=False,
    ).drop_duplicates(subset=["flow_uid"])
    merged = result.merge(tuple_df, on="flow_uid", how="left")

    front_columns = ["flow_uid", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
    other_columns = [col for col in merged.columns if col not in front_columns]
    merged = merged[front_columns + other_columns]

    predictions_csv = output_dir / "predictions.csv"
    merged.to_csv(predictions_csv, index=False, encoding="utf-8")

    pred_counts = merged["pred_label_name"].value_counts().to_dict()
    malicious = int(sum(count for name, count in pred_counts.items() if name != "benign"))
    total = int(len(merged))
    summary = {
        "pcap_filename": pcap_name,
        "total_flows": total,
        "pred_counts": {name: int(count) for name, count in pred_counts.items()},
        "malicious_flows": malicious,
        "malicious_ratio": round(malicious / total, 4) if total else 0.0,
        "stage_timings_sec": {str(stage): round(sec, 2) for stage, sec in timings.items()},
        "models": {
            "model_release": ACTIVE_RELEASE_ID,
            "encoder": str(PRODUCTION_PRETRAIN_DIR),
            "lora_adapter": str(PRODUCTION_LORA_DIR / "best"),
            "supcon_ae": str(SUPCON_MODEL_PATH),
            "lightgbm": str(LGB_MODEL_PKL),
        },
        "predictions_csv": str(predictions_csv),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_inference(pcap_path: str | Path, output_dir: str | Path, batch_size: int = 16) -> dict:
    pcap_path = Path(pcap_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP 文件不存在: {pcap_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    load_active_bundle()
    missing = [str(path) for path in required_checkpoints() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "已发布模型不完整。请先由开发者训练并发布 models/production 中的新版本:\n  " + "\n  ".join(missing)
        )

    # process_one_file writes temp CSVs relative to CWD; keep them inside the task dir.
    os.chdir(output_dir)

    started = time.time()
    timings: dict[int, float] = {}
    print(f"[推理] 输入: {pcap_path}")
    print(f"[推理] 输出: {output_dir}")

    emit_stage(1)
    t0 = time.time()
    # The canonical extractor consumes the original capture. Per-flow truncation before TCP
    # reassembly would discard retransmission/ordering evidence and can cut TLS
    # records in half.
    timings[1] = time.time() - t0
    emit_stage_done(1, {"duration_sec": round(timings[1], 2), "capture": "validated by Scapy reader"})

    emit_stage(2)
    t0 = time.time()
    features_csv, temporal_csv = stage_flow_features(pcap_path, output_dir)
    timings[2] = time.time() - t0
    emit_stage_done(2, {"duration_sec": round(timings[2], 2)})
    flow_count = sum(1 for _ in open(features_csv, encoding="utf-8")) - 1
    if flow_count <= 0:
        raise RuntimeError("PCAP 中未提取到任何双向流，无法推理")

    emit_stage(3)
    t0 = time.time()
    jsonl_path = stage_serialize(features_csv, output_dir)
    timings[3] = time.time() - t0
    emit_stage_done(3, {"duration_sec": round(timings[3], 2)})

    emit_stage(4)
    t0 = time.time()
    flows, semantic = stage_semantic_features(jsonl_path, batch_size)
    timings[4] = time.time() - t0
    emit_stage_done(4, {"duration_sec": round(timings[4], 2)})

    emit_stage(5)
    t0 = time.time()
    result = stage_predict(flows, semantic)
    timings[5] = time.time() - t0

    summary = stage_write_outputs(result, features_csv, output_dir, pcap_path.name, timings)
    emit_stage_done(5, {"duration_sec": round(timings[5], 2)})

    summary["elapsed_sec"] = round(time.time() - started, 2)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    emit_task_done(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="单文件纯推理: PCAP -> 八类加密流量预测")
    parser.add_argument("--pcap", required=True, help="输入 pcap/pcapng 路径")
    parser.add_argument("--output-dir", required=True, help="任务输出目录")
    parser.add_argument("--batch-size", type=int, default=16, help="DeBERTa 推理 batch 大小")
    args = parser.parse_args()
    run_inference(args.pcap, args.output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
