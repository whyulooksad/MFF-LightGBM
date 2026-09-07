# -*- coding: utf-8 -*-
"""Measure resource cost, latency, and throughput for pipeline stages.

The benchmark is intentionally separate from the offline CLI so normal experiments do
not accidentally overwrite results or spend extra time collecting metrics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil

from developer.config import (
    EVALUATION_BENCHMARKS_DIR,
    EXPERIMENT_SUPCON_MODEL_PATH,
    EXPERIMENT_LORA_DIR,
    EXPERIMENT_PRETRAIN_DIR,
    DATASET_PREPARED_DIR,
    FEATURE_FLOWS_JSONL,
    FEATURES_FUSED_CSV,
    FEATURES_PURE_CSV,
    FLOW_FEATURES_ALL_CSV,
    PIPELINE_OUTPUT_DIR,
    ROOT,
)


REPORT_DIR = EVALUATION_BENCHMARKS_DIR / "resource_benchmark"
REPORT_CSV = REPORT_DIR / "resource_benchmark.csv"
SUMMARY_JSON = REPORT_DIR / "resource_benchmark_summary.json"


@dataclass
class StageMetric:
    stage: str
    status: str
    input_path: str
    output_path: str
    input_rows: int | None
    output_rows: int | None
    wall_time_s: float | None
    cpu_time_s: float | None
    peak_rss_mb: float | None
    peak_gpu_mb: float | None
    throughput_rows_s: float | None
    note: str = ""


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def first_existing_pcap() -> Path | None:
    return next(
        (path for path in DATASET_PREPARED_DIR.rglob("*") if path.suffix.lower() in {".pcap", ".pcapng"}),
        None,
    )


def reset_gpu_peak() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except Exception:
        return False
    return False


def gpu_peak_mb(enabled: bool) -> float | None:
    if not enabled:
        return None
    try:
        import torch

        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


@contextmanager
def measured_context():
    gc.collect()
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    start_cpu = process.cpu_times()
    start_time = time.perf_counter()
    gpu_enabled = reset_gpu_peak()
    try:
        yield
    finally:
        pass
    end_time = time.perf_counter()
    end_cpu = process.cpu_times()
    end_mem = process.memory_info().rss
    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    wall_time = end_time - start_time
    peak_rss = max(start_mem, end_mem) / (1024 * 1024)
    peak_gpu = gpu_peak_mb(gpu_enabled)
    measured_context.wall_time = wall_time
    measured_context.cpu_time = cpu_time
    measured_context.peak_rss = peak_rss
    measured_context.peak_gpu = peak_gpu


def run_stage(
    stage: str,
    input_path: Path,
    output_path: Path,
    input_rows: int | None,
    output_counter: Callable[[Path], int],
    runner: Callable[[], None],
    note: str = "",
) -> StageMetric:
    try:
        with measured_context():
            runner()
        output_rows = output_counter(output_path)
        wall_time = measured_context.wall_time
        throughput = output_rows / wall_time if output_rows and wall_time > 0 else None
        return StageMetric(
            stage=stage,
            status="ok",
            input_path=str(input_path),
            output_path=str(output_path),
            input_rows=input_rows,
            output_rows=output_rows,
            wall_time_s=round(wall_time, 4),
            cpu_time_s=round(measured_context.cpu_time, 4),
            peak_rss_mb=round(measured_context.peak_rss, 2),
            peak_gpu_mb=round(measured_context.peak_gpu, 2) if measured_context.peak_gpu is not None else None,
            throughput_rows_s=round(throughput, 4) if throughput is not None else None,
            note=note,
        )
    except Exception as exc:
        return StageMetric(
            stage=stage,
            status="failed",
            input_path=str(input_path),
            output_path=str(output_path),
            input_rows=input_rows,
            output_rows=None,
            wall_time_s=None,
            cpu_time_s=None,
            peak_rss_mb=None,
            peak_gpu_mb=None,
            throughput_rows_s=None,
            note=f"{type(exc).__name__}: {exc}",
        )


def measure_manual_feature_extraction() -> StageMetric:
    pcap = first_existing_pcap()
    sample_output = REPORT_DIR / "manual_features_sample.csv"
    if pcap is None:
        return StageMetric(
            stage="manual_feature_extraction",
            status="skipped",
            input_path=str(DATASET_PREPARED_DIR),
            output_path=str(sample_output),
            input_rows=None,
            output_rows=0,
            wall_time_s=None,
            cpu_time_s=None,
            peak_rss_mb=None,
            peak_gpu_mb=None,
            throughput_rows_s=None,
            note=(
                "Missing prepared PCAP/PCAPNG under data/developer/datasets/prepared."
            ),
        )

    def runner() -> None:
        from user_app.inference.extract_flow_features import CSV_HEADERS, extract_pcap, write_csv
        rows, _ = extract_pcap(pcap)
        write_csv(sample_output, rows, CSV_HEADERS)

    return run_stage(
        stage="manual_feature_extraction",
        input_path=pcap,
        output_path=sample_output,
        input_rows=None,
        output_counter=count_csv_rows,
        runner=runner,
        note="PCAP/PCAPNG -> flow feature CSV.",
    )


def measure_llm_feature_extraction(max_samples: int | None) -> StageMetric:
    from user_app.inference.semantic_encoder import default_lora_adapter, extract_features, latest_pretrain_checkpoint
    from user_app.inference.serialize_flow import preprocess_feature

    nrows_note = f"max_samples={max_samples}" if max_samples else "full dataset"

    def runner() -> None:
        preprocess_feature(str(FLOW_FEATURES_ALL_CSV), str(FEATURE_FLOWS_JSONL), nrows=max_samples)
        extract_features(
            base_model_dir=latest_pretrain_checkpoint(EXPERIMENT_PRETRAIN_DIR),
            adapter_dir=default_lora_adapter(EXPERIMENT_LORA_DIR),
            jsonl_path=str(FEATURE_FLOWS_JSONL),
            output_pure_csv=str(FEATURES_PURE_CSV),
            output_fused_csv=str(FEATURES_FUSED_CSV),
            max_samples=max_samples,
        )

    return run_stage(
        stage="llm_feature_extraction",
        input_path=FLOW_FEATURES_ALL_CSV,
        output_path=FEATURES_FUSED_CSV,
        input_rows=count_csv_rows(FLOW_FEATURES_ALL_CSV),
        output_counter=count_csv_rows,
        runner=runner,
        note=f"Flow feature CSV -> JSONL -> LLM pure/fused feature CSV; {nrows_note}.",
    )


def detector_inference_runner(max_samples: int | None) -> Callable[[], None]:
    def runner() -> None:
        import pickle

        from developer.detector.train_lightgbm import (
            FEATURE_COLUMNS_JSON,
            INPUT_CSV,
            MODEL_PKL,
            detector_feature_columns,
            predict_in_batches,
            preprocess_detector_dataframe,
        )
        from user_app.inference.supcon_reducer import SupConAEReducer, replace_semantic_features

        raw_df = pd.read_csv(INPUT_CSV, nrows=max_samples)
        reducer = SupConAEReducer.load(EXPERIMENT_SUPCON_MODEL_PATH, verbose=False)
        raw_df = replace_semantic_features(raw_df, reducer)
        df = preprocess_detector_dataframe(raw_df)
        if FEATURE_COLUMNS_JSON.exists():
            feature_cols = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
        else:
            feature_cols = detector_feature_columns(df)
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing detector feature columns: {missing[:5]}")
        X = df[feature_cols].values.astype(np.float32)
        with MODEL_PKL.open("rb") as f:
            model = pickle.load(f)
        proba = predict_in_batches(model, X)
        pred = np.argmax(proba, axis=1)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"pred_label": pred})
        if "flow_uid" in raw_df.columns:
            out.insert(0, "flow_uid", raw_df["flow_uid"].values)
        out.to_csv(REPORT_DIR / "detector_inference_predictions.csv", index=False)

    return runner


def measure_detector_inference(max_samples: int | None) -> StageMetric:
    output_path = REPORT_DIR / "detector_inference_predictions.csv"
    nrows_note = f"max_samples={max_samples}" if max_samples else "full dataset"
    input_rows = min(max_samples, count_csv_rows(FEATURES_FUSED_CSV)) if max_samples else count_csv_rows(FEATURES_FUSED_CSV)
    return run_stage(
        stage="detector_inference",
        input_path=FEATURES_FUSED_CSV,
        output_path=output_path,
        input_rows=input_rows,
        output_counter=count_csv_rows,
        runner=detector_inference_runner(max_samples),
        note=f"Trained LightGBM detector inference only, no retraining; {nrows_note}.",
    )


def write_reports(metrics: list[StageMetric]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for row in metrics:
            writer.writerow(asdict(row))

    ok_metrics = [m for m in metrics if m.status == "ok" and m.wall_time_s is not None]
    summary = {
        "project_root": str(ROOT),
        "prepared_pcaps": sum(1 for path in DATASET_PREPARED_DIR.rglob("*") if path.suffix.lower() in {".pcap", ".pcapng"}),
        "total_wall_time_s": round(sum(m.wall_time_s for m in ok_metrics), 4),
        "total_cpu_time_s": round(sum(m.cpu_time_s or 0 for m in ok_metrics), 4),
        "stages": [asdict(m) for m in metrics],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pipeline resource, latency, and throughput.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for LLM extraction and detector inference.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip expensive LLM feature extraction and only benchmark detector inference.",
    )
    args = parser.parse_args()

    metrics = [measure_manual_feature_extraction()]
    if not args.skip_llm:
        metrics.append(measure_llm_feature_extraction(args.max_samples))
    metrics.append(measure_detector_inference(args.max_samples))

    write_reports(metrics)
    print(pd.DataFrame([asdict(m) for m in metrics]).to_string(index=False))
    print(f"\nreport_csv={REPORT_CSV}")
    print(f"summary_json={SUMMARY_JSON}")


if __name__ == "__main__":
    main()
