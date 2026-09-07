"""Developer-only offline model preparation and evaluation entry point.

This command is never used for a user upload. The Web application calls
``user_app.inference.pipeline`` and loads frozen production models.
"""

from __future__ import annotations

import argparse
import runpy


DEFAULT_PIPELINE_STAGES = ["flow_features", "preprocess", "pretrain", "lora", "extract", "supcon", "detector"]
ALL_PIPELINE_STAGES = [*DEFAULT_PIPELINE_STAGES, "evaluate"]


def run_flow_features():
    from developer.data_prepare.batch_extract import batch_extract
    from developer.config import DATASET_PREPARED_DIR, DATASET_SPLIT_MANIFEST, FLOW_FEATURES_DIR

    batch_extract(DATASET_PREPARED_DIR, FLOW_FEATURES_DIR, DATASET_SPLIT_MANIFEST)


def run_preprocess():
    from developer.config import (
        FEATURE_FLOWS_JSONL, FLOW_FEATURES_ALL_CSV, PRETRAIN_CORPUS_DIR, SUPERVISED_CORPUS_DIR,
    )
    from user_app.inference.serialize_flow import preprocess_feature, preprocess_pretrain, preprocess_supervised

    source = str(FLOW_FEATURES_ALL_CSV)
    preprocess_pretrain(source, str(PRETRAIN_CORPUS_DIR / "pretrain_flows.jsonl"))
    preprocess_supervised(source, str(SUPERVISED_CORPUS_DIR / "supervised_flows.jsonl"))
    preprocess_feature(source, str(FEATURE_FLOWS_JSONL))


def run_pretrain():
    runpy.run_module("developer.representation.pretrain", run_name="__main__")


def run_lora():
    runpy.run_module("developer.representation.train_lora", run_name="__main__")


def run_extract():
    from developer.config import (
        EXPERIMENT_LORA_DIR, EXPERIMENT_PRETRAIN_DIR, FEATURE_FLOWS_JSONL,
        FEATURES_FUSED_CSV, FEATURES_PURE_CSV,
    )
    from user_app.inference.semantic_encoder import extract_features

    from user_app.inference.semantic_encoder import default_lora_adapter, latest_pretrain_checkpoint

    extract_features(
        base_model_dir=latest_pretrain_checkpoint(EXPERIMENT_PRETRAIN_DIR),
        adapter_dir=default_lora_adapter(EXPERIMENT_LORA_DIR),
        jsonl_path=str(FEATURE_FLOWS_JSONL),
        output_pure_csv=str(FEATURES_PURE_CSV),
        output_fused_csv=str(FEATURES_FUSED_CSV),
    )


def run_supcon():
    runpy.run_module("developer.detector.train_supcon", run_name="__main__")


def run_detector():
    from developer.detector.train_lightgbm import main

    main()


def run_evaluate():
    runpy.run_module("developer.evaluation.feature_ablation", run_name="__main__")


STAGE_RUNNERS = {
    "flow_features": run_flow_features,
    "preprocess": run_preprocess,
    "pretrain": run_pretrain,
    "lora": run_lora,
    "extract": run_extract,
    "supcon": run_supcon,
    "detector": run_detector,
    "evaluate": run_evaluate,
}


def main():
    parser = argparse.ArgumentParser(description="Run the developer-only offline training/evaluation pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", *ALL_PIPELINE_STAGES],
        default="all",
        help="Pipeline stage to run. Default: all.",
    )
    args = parser.parse_args()

    stages = list(DEFAULT_PIPELINE_STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n=== Pipeline stage: {stage} ===")
        STAGE_RUNNERS[stage]()


if __name__ == "__main__":
    main()
