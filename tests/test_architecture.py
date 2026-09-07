from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_user_app_does_not_import_developer_channel():
    violations = []
    for path in (ROOT / "user_app").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "from developer" in source or "import developer" in source:
            violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_removed_package_name_is_not_reintroduced():
    violations = []
    for root_name in ("developer", "user_app"):
        for path in (ROOT / root_name).rglob("*.py"):
            if "mff_lightgbm" in path.read_text(encoding="utf-8"):
                violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_formal_code_never_imports_legacy_archive():
    violations = []
    for root_name in ("developer", "user_app"):
        for path in (ROOT / root_name).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "from legacy" in source or "import legacy" in source:
                violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_user_backend_has_no_developer_data_source():
    violations = []
    for path in (ROOT / "user_app" / "backend").rglob("*.py"):
        if "data/developer" in path.read_text(encoding="utf-8").replace("\\", "/"):
            violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_approved_top_level_code_structure_exists():
    required = [
        "developer/data_prepare/build_manifest.py",
        "developer/data_prepare/map_labels.py",
        "developer/data_prepare/split_dataset.py",
        "developer/data_prepare/batch_extract.py",
        "developer/representation/config.py",
        "developer/detector/train_supcon.py",
        "developer/evaluation/evaluate.py",
        "developer/evaluation/compare_versions.py",
        "developer/release/validate_bundle.py",
        "developer/release/build_contract.py",
        "developer/release/publish.py",
        "user_app/inference/pcap_reader.py",
        "user_app/inference/tcp_reassembly.py",
        "user_app/inference/tls_parser.py",
        "user_app/inference/x509_parser.py",
        "user_app/inference/extract_flow_features.py",
        "user_app/inference/lightgbm_detector.py",
        "user_app/inference/model_loader.py",
    ]
    assert [path for path in required if not (ROOT / path).is_file()] == []
