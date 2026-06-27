"""Entry point for the final encrypted-traffic detection pipeline."""

from __future__ import annotations

import argparse
import runpy


DEFAULT_PIPELINE_STAGES = ["truncate", "flow_features", "preprocess", "extract", "detector"]
ALL_PIPELINE_STAGES = ["truncate", "flow_features", "preprocess", "extract", "supcon", "detector"]


def run_truncate():
    runpy.run_module("pipeline.truncate_pcap", run_name="__main__")


def run_flow_features():
    from pipeline.extract_flow_features import main

    main()


def run_preprocess():
    from preprocess import preprocess_feature

    preprocess_feature()


def run_extract():
    from pipeline.LLM_extract_features import extract_features

    extract_features()


def run_supcon():
    runpy.run_module("pipeline.supcon_ae", run_name="__main__")


def run_detector():
    from pipeline.detector import main

    main()


STAGE_RUNNERS = {
    "truncate": run_truncate,
    "flow_features": run_flow_features,
    "preprocess": run_preprocess,
    "extract": run_extract,
    "supcon": run_supcon,
    "detector": run_detector,
}


def main():
    parser = argparse.ArgumentParser(description="Run the encrypted-traffic detection pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", *ALL_PIPELINE_STAGES],
        default="all",
        help="Pipeline stage to run. Default: all.",
    )
    parser.add_argument(
        "--skip-truncate",
        action="store_true",
        help="When --stage all is used, start from existing pcap files in data/pcap/truncated.",
    )
    parser.add_argument(
        "--use-supcon",
        action="store_true",
        help="When --stage all is used, run optional SupCon-AE between feature extraction and detector.",
    )
    args = parser.parse_args()

    stages = list(DEFAULT_PIPELINE_STAGES) if args.stage == "all" else [args.stage]
    if args.stage == "all" and args.use_supcon:
        stages.insert(stages.index("detector"), "supcon")
    if args.skip_truncate and args.stage == "all":
        stages = [stage for stage in stages if stage != "truncate"]

    for stage in stages:
        print(f"\n=== Pipeline stage: {stage} ===")
        STAGE_RUNNERS[stage]()


if __name__ == "__main__":
    main()
