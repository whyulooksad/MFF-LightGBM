"""Shared paths for the final detection pipeline."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PCAP_RAW_DIR = DATA_DIR / "pcap" / "raw"
PCAP_TRUNCATED_DIR = DATA_DIR / "pcap" / "truncated"

FLOW_FEATURES_DIR = DATA_DIR / "flow_features"
FLOW_FEATURES_TRAIN_CSV = FLOW_FEATURES_DIR / "final_multiclass_features_train.csv"
FLOW_FEATURES_TEST_CSV = FLOW_FEATURES_DIR / "final_multiclass_features_test.csv"
FLOW_METADATA_TEMPORAL_TRAIN_CSV = FLOW_FEATURES_DIR / "flow_metadata_temporal_train.csv"
FLOW_METADATA_TEMPORAL_TEST_CSV = FLOW_FEATURES_DIR / "flow_metadata_temporal_test.csv"

PIPELINE_DATA_DIR = DATA_DIR / "pipeline"
PIPELINE_INPUT_DIR = PIPELINE_DATA_DIR / "input"
PIPELINE_FEATURES_DIR = PIPELINE_DATA_DIR / "features"
PIPELINE_OUTPUT_DIR = PIPELINE_DATA_DIR / "output"
DETECTION_ASSETS_DIR = PIPELINE_OUTPUT_DIR / "detection_assets"

FEATURE_FLOWS_JSONL = PIPELINE_INPUT_DIR / "feature_flows.jsonl"
FEATURES_PURE_CSV = PIPELINE_FEATURES_DIR / "features_pure.csv"
FEATURES_FUSED_CSV = PIPELINE_FEATURES_DIR / "features_fused.csv"
REDUCED_FEATURES_CSV = PIPELINE_FEATURES_DIR / "reduced_features.csv"
DETECTION_RESULTS_CSV = PIPELINE_OUTPUT_DIR / "detection_results.csv"

MODELS_DIR = ROOT / "models"
DETECTOR_MODEL_DIR = MODELS_DIR / "detector"
SUPCON_MODEL_PATH = ROOT / "checkpoints" / "supcon_ae" / "demo_supcon_ae.pt"
