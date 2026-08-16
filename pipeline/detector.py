# -*- coding: utf-8 -*-
"""LightGBM detector stage: train, validate, test, and export reports."""

from __future__ import annotations

import json
import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from LLM_train.config import ID2LABEL, LABEL2ID
from pipeline.config import (
    DETECTION_ASSETS_DIR,
    DETECTION_RESULTS_CSV,
    FEATURES_FUSED_CSV,
    PIPELINE_OUTPUT_DIR,
    ROOT,
    REDUCED_FEATURES_CSV,
    SUPCON_MODEL_PATH,
)
from pipeline.pca_reduce_features import reduce_feat_in_memory
from pipeline.supcon_ae import SupConAEReducer, replace_semantic_features, semantic_feature_columns


warnings.filterwarnings("ignore")

INPUT_CSV = FEATURES_FUSED_CSV
OUTPUT_CSV = DETECTION_RESULTS_CSV
OUTPUT_DIR = PIPELINE_OUTPUT_DIR
ASSETS_DIR = DETECTION_ASSETS_DIR
CHECKPOINT_DIR = ROOT / "checkpoints" / "detector"
MODEL_PKL = CHECKPOINT_DIR / "best_lgb_model.pkl"
MODEL_TXT = CHECKPOINT_DIR / "best_lgb_model.txt"
FEATURE_COLUMNS_JSON = CHECKPOINT_DIR / "feature_columns.json"
REPORT_DIR = OUTPUT_DIR / "detector_report"

LABEL_COL = "label"
ID_COL_CANDIDATES = ("flow_uid", "flow_id")
DROP_COLS = (
    "flow_uid",
    "flow_id",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "timestamp",
    "dataset_source",
    "subfolder",
    "pcap_filename",
    "zeek_conn_log",
    "zeek_ssl_log",
    "zeek_x509_log",
    "label_name",
)

TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.125
SEED = 42
BATCH_SIZE = 4096
BENIGN_LABEL = 0
MALICIOUS_CONF_THRESHOLD = 0.8

CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "pastel_blues",
    ["#fbfdff", "#e4f0f8", "#bdd9ed", "#79acd0", "#2f6f9f"],
)
METRIC_PALETTE = {
    "Accuracy": "#FF8C00",
    "Precision": "#1E90FF",
    "Recall": "#00BFFF",
    "Weighted F1": "#9B59B6",
}


def id_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in ID_COL_CANDIDATES if col in df.columns]


def normalize_label_series(labels: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(labels, errors="coerce")
    missing_numeric = numeric.isna() & labels.notna()
    if not missing_numeric.any():
        return numeric.astype("Int64")

    normalized = labels.astype(str).str.strip().str.lower()
    mapped = normalized.map(LABEL2ID)
    unknown = sorted(normalized[mapped.isna() & labels.notna()].unique())
    if unknown:
        raise ValueError(f"Unknown detector labels: {unknown}. Expected one of {sorted(LABEL2ID)}")
    return mapped.astype("Int64")


def extract_cn_features(df: pd.DataFrame) -> pd.DataFrame:
    if "cn_value" not in df.columns:
        return pd.DataFrame(index=df.index)

    cn = df["cn_value"].fillna("").astype(str)
    feats = pd.DataFrame(index=df.index)
    feats["cn_len"] = cn.str.len()

    tld = cn.str.split(".").str[-1].str.lower()
    tld_counts = tld.value_counts()
    common_tlds = tld_counts[tld_counts >= 5].index.tolist()
    tld = tld.apply(lambda value: value if value in common_tlds else "other")
    feats["cn_tld_encoded"] = LabelEncoder().fit_transform(tld)

    def count_subdomains(domain: str) -> int:
        parts = domain.split(".")
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return max(0, len(parts) - 1)

    feats["cn_subdomain_count"] = cn.apply(count_subdomains)

    ip_pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
    feats["cn_has_ip"] = cn.apply(lambda value: 1 if ip_pattern.match(value) else 0)

    def calc_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        probs = np.array(list(counts.values())) / len(text)
        return float(entropy(probs, base=2))

    feats["cn_entropy"] = cn.apply(calc_entropy)
    feats["cn_is_www"] = cn.str.startswith("www.").astype(int)
    feats["cn_digit_ratio_new"] = cn.apply(lambda text: sum(ch.isdigit() for ch in text) / max(len(text), 1))
    feats["cn_has_hyphen"] = cn.str.contains("-").astype(int)
    return feats


def preprocess_detector_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LABEL_COL in df.columns:
        df[LABEL_COL] = normalize_label_series(df[LABEL_COL])

    protected = set(id_columns(df) + ([LABEL_COL] if LABEL_COL in df.columns else []))
    drop_cols = [col for col in DROP_COLS if col in df.columns and col not in protected]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cn_feats = extract_cn_features(df)
    if not cn_feats.empty:
        df = pd.concat([df, cn_feats], axis=1)
        if "cn_value" in df.columns:
            df = df.drop(columns=["cn_value"])

    non_num_cols = df.select_dtypes(include=["object"]).columns.tolist()
    non_num_cols = [col for col in non_num_cols if col not in protected]
    for col in non_num_cols:
        if df[col].nunique(dropna=False) < 50:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df = df.drop(columns=[col])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols and df[num_cols].isnull().any().any():
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    return df


def detector_feature_columns(df: pd.DataFrame) -> list[str]:
    protected = set(id_columns(df) + ([LABEL_COL] if LABEL_COL in df.columns else []))
    return [
        col
        for col in df.columns
        if col not in protected and pd.api.types.is_numeric_dtype(df[col])
    ]


def save_feature_columns(feature_columns: list[str]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_JSON.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_lgb_model(model: lgb.Booster) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)

    with open(MODEL_PKL, "rb") as f:
        loaded = pickle.load(f)
    probe = np.random.rand(1, model.num_feature()).astype(np.float32)
    loaded.predict(probe)

    try:
        model.save_model(str(MODEL_TXT), num_iteration=model.best_iteration)
    except Exception as exc:
        print(f"[WARN] Could not save LightGBM text backup: {exc}")


def predict_in_batches(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    chunks = []
    best_iteration = getattr(model, "best_iteration", None)
    for start in range(0, len(X), BATCH_SIZE):
        batch = X[start : start + BATCH_SIZE]
        if best_iteration:
            proba = model.predict(batch, num_iteration=best_iteration)
        else:
            proba = model.predict(batch)
        chunks.append(proba)
    return np.vstack(chunks)


def metrics_row(name: str, y_true, y_pred) -> list[float | str]:
    return [
        name,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="weighted"),
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="weighted", zero_division=0),
    ]


def write_test_outputs(test_df: pd.DataFrame, y_test: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)

    ids = id_columns(test_df)
    out_df = test_df[ids].copy() if ids else pd.DataFrame(index=test_df.index)
    out_df["true_label"] = y_test
    out_df["true_label_name"] = pd.Series(y_test).map(ID2LABEL).fillna(pd.Series(y_test).astype(str))
    out_df["pred_label"] = pred
    out_df["pred_label_name"] = pd.Series(pred).map(ID2LABEL).fillna(pd.Series(pred).astype(str))
    out_df["confidence"] = conf

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    out_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    labels = sorted(ID2LABEL)
    cm = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(
        cm,
        index=[ID2LABEL[idx] for idx in labels],
        columns=[ID2LABEL[idx] for idx in labels],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    plot_confusion_matrix(cm, labels, y_test, pred)

    report = classification_report(
        y_test,
        pred,
        labels=labels,
        target_names=[ID2LABEL[idx] for idx in labels],
        digits=4,
        zero_division=0,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return out_df


def plot_confusion_matrix(cm: np.ndarray, labels: list[int], y_test: np.ndarray, pred: np.ndarray) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    weighted_f1 = f1_score(y_test, pred, average="weighted")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_for_color = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(7.2, 5.8))
    ax = sns.heatmap(
        cm_for_color,
        annot=cm,
        annot_kws={"fontsize": 10},
        fmt="d",
        cmap=CONFUSION_CMAP,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3,
        linecolor="white",
        vmin=0,
        vmax=1,
    )

    # 单独加深左上角 cell (benign x benign),让它比对角线其他格子更突出
    from matplotlib.patches import Rectangle

    ylim = ax.get_ylim()
    top = ylim[1]
    y0 = top if ylim[0] > ylim[1] else top - 1
    ax.add_patch(
        Rectangle(
            (0, y0), 1, 1,
            facecolor="#173f5a", edgecolor="white", linewidth=0.3, zorder=2,
        )
    )
    ax.text(
        0.5, y0 + 0.5, str(int(cm[0, 0])),
        ha="center", va="center", fontsize=10, color="white", zorder=3,
    )

    plt.title(f"LightGBM (Weighted F1={weighted_f1:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def write_detection_assets(raw_test_df: pd.DataFrame, out_df: pd.DataFrame, proba: np.ndarray) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    pred = out_df["pred_label"].to_numpy()
    summary = {
        "total_flows": len(out_df),
        "predicted_benign": int(np.sum(pred == BENIGN_LABEL)),
        "predicted_malicious": int(np.sum(pred != BENIGN_LABEL)),
    }
    for label, count in pd.Series(pred).value_counts().sort_index().items():
        summary[f"class_{label}_count"] = int(count)
    pd.DataFrame([summary]).to_csv(ASSETS_DIR / "detection_summary.csv", index=False)

    risk_df = out_df[id_columns(out_df)].copy() if id_columns(out_df) else pd.DataFrame(index=out_df.index)
    risk_df["confidence"] = out_df["confidence"]
    risk_df["pred_label"] = out_df["pred_label"]
    risk_df["pred_label_name"] = out_df["pred_label_name"]
    risk_df.to_csv(ASSETS_DIR / "risk_distribution.csv", index=False)

    high_conf_mal = (out_df["pred_label"] != BENIGN_LABEL) & (
        out_df["confidence"] >= MALICIOUS_CONF_THRESHOLD
    )
    details = pd.concat(
        [
            raw_test_df.loc[high_conf_mal].reset_index(drop=True),
            out_df.loc[high_conf_mal].reset_index(drop=True),
        ],
        axis=1,
    )
    details.to_csv(ASSETS_DIR / "malicious_details.csv", index=False)

    trend_rows = []
    window = 100
    for start in range(0, len(out_df), window):
        end = min(start + window, len(out_df))
        segment = pred[start:end]
        trend_rows.append(
            {
                "sample_start": start,
                "sample_end": end - 1,
                "total": len(segment),
                "malicious": int(np.sum(segment != BENIGN_LABEL)),
            }
        )
    pd.DataFrame(trend_rows).to_csv(ASSETS_DIR / "trend_data.csv", index=False)

    feat_cols = [col for col in raw_test_df.columns if col.startswith("feat_")]
    if feat_cols and len(raw_test_df) >= 2:
        coords = PCA(n_components=2, random_state=SEED).fit_transform(
            raw_test_df[feat_cols].fillna(0).to_numpy(np.float32)
        )
        latent = out_df[id_columns(out_df)].copy() if id_columns(out_df) else pd.DataFrame(index=out_df.index)
        latent["x"] = coords[:, 0]
        latent["y"] = coords[:, 1]
        latent["pred_label"] = pred
        latent.to_csv(ASSETS_DIR / "latent_2d.csv", index=False)

    pd.DataFrame(proba, columns=[f"prob_{idx}" for idx in range(proba.shape[1])]).to_csv(
        ASSETS_DIR / "class_probabilities.csv",
        index=False,
    )


def write_weighted_metrics_plot(results_df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = results_df.rename(columns={"F1": "Weighted F1"}).melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "Weighted F1"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(13, 6.5))
    ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette=METRIC_PALETTE)
    ax.set_ylim(0, 1.05)
    ax.set_title("Weighted Metrics Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "weighted_metrics_comparison.png", dpi=200)
    plt.close()


def write_roc_report(y_test: np.ndarray, proba_by_model: dict[str, np.ndarray]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    classes = np.array(sorted(ID2LABEL))
    y_bin = label_binarize(y_test, classes=classes)
    auc_rows = []

    plt.figure(figsize=(10, 7))
    for name, proba in proba_by_model.items():
        if proba is None or proba.shape != y_bin.shape:
            continue
        if len(np.unique(y_bin.ravel())) <= 1:
            continue

        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        auc_value = roc_auc_score(y_bin, proba, average="micro", multi_class="ovr")
        auc_rows.append({"model": name, "micro_auc": auc_value})
        linewidth = 2.8 if name == "LightGBM (ours)" else 1.8
        linestyle = "--" if name == "LightGBM (ours)" else "-"
        plt.plot(fpr, tpr, linewidth=linewidth, linestyle=linestyle, label=f"{name} micro AUC={auc_value:.3f}")

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Classifier Micro-Average ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=200)
    plt.close()

    pd.DataFrame(auc_rows).to_csv(REPORT_DIR / "roc_auc_scores.csv", index=False)


def evaluate_baselines(
    X_train,
    X_test,
    y_train,
    y_test,
    lgb_pred,
    lgb_proba,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    results = [metrics_row("LightGBM (ours)", y_test, lgb_pred)]
    proba_by_model = {"LightGBM (ours)": lgb_proba}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baselines = {
        "KNN (k=5)": (KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, random_state=SEED),
            X_train,
            X_test,
        ),
        "Logistic Regression": (
            LogisticRegression(solver="saga", max_iter=300, n_jobs=-1, random_state=SEED),
            X_train_scaled,
            X_test_scaled,
        ),
        "Gaussian NB": (GaussianNB(), X_train, X_test),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
            X_train,
            X_test,
        ),
    }
    if XGB_AVAILABLE:
        baselines["XGBoost"] = (
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=SEED,
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=-1,
            ),
            X_train,
            X_test,
        )
    else:
        print("  [WARN] XGBoost is not installed; skip XGBoost baseline")

    for name, (baseline, train_x, test_x) in baselines.items():
        baseline.fit(train_x, y_train)
        y_pred = baseline.predict(test_x)
        results.append(metrics_row(name, y_test, y_pred))
        if hasattr(baseline, "predict_proba"):
            proba_by_model[name] = baseline.predict_proba(test_x)

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "Precision", "Recall"])
    return results_df.sort_values(by="F1", ascending=False), proba_by_model


def main(
    n_components: int = 64,
    reducer_name: str = "supcon",
    supcon_config: dict | None = None,
    run_baselines: bool = True,
):
    print("=" * 60)
    print(" LightGBM detector: train / validate / test")
    print("=" * 60)

    print(f"\n[1/6] Read fused features: {INPUT_CSV}")
    raw_df = pd.read_csv(INPUT_CSV)
    df = preprocess_detector_dataframe(raw_df)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Detector stage requires a '{LABEL_COL}' column for train/test split")
    y = df[LABEL_COL].astype("int64").to_numpy()
    print(f"  samples={len(df):,}, raw_features={len(detector_feature_columns(df))}, classes={len(np.unique(y))}")

    print("\n[2/6] Split train / validation / test before supervised reduction")
    indices = np.arange(len(df))
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        indices,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    train_idx, val_idx, y_train_lgb, y_val_lgb = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        random_state=SEED,
        stratify=y_train_val,
    )

    print("\n[3/6] Reduce DeBERTa semantic features")
    if reducer_name == "supcon":
        semantic_cols = semantic_feature_columns(df)
        if not semantic_cols:
            raise ValueError("No feat_* semantic columns found for SupCon-AE")
        semantic_values = df[semantic_cols].to_numpy(np.float32)
        reducer = SupConAEReducer(config=supcon_config, verbose=True)
        reducer.fit(
            semantic_values[train_idx],
            y[train_idx],
            semantic_values[val_idx],
            y[val_idx],
            feature_columns=semantic_cols,
            checkpoint_path=SUPCON_MODEL_PATH,
        )
        df = replace_semantic_features(df, reducer)
        df.to_csv(REDUCED_FEATURES_CSV, index=False)
        print(
            f"  SupCon-AE: {len(semantic_cols)} -> {reducer.config['latent_dim']} dimensions; "
            f"checkpoint={SUPCON_MODEL_PATH}"
        )
        print(f"  reduced_features={REDUCED_FEATURES_CSV}")
    elif reducer_name == "pca":
        print(f"  PCA: feat_* -> {n_components} dimensions")
        df = reduce_feat_in_memory(df, n_components=n_components, seed=SEED)
    elif reducer_name != "none":
        raise ValueError(f"Unknown reducer: {reducer_name}")

    feature_cols = detector_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric detector features found")
    save_feature_columns(feature_cols)
    X = df[feature_cols].values.astype(np.float32)
    print(f"  detector_features={X.shape[1]}")

    X_train_lgb = X[train_idx]
    X_val_lgb = X[val_idx]
    X_test = X[test_idx]
    X_train_baseline = X[train_val_idx]

    print(
        "  split ratio: "
        f"train={len(train_idx) / len(df):.1%}, "
        f"val={len(val_idx) / len(df):.1%}, "
        f"test={len(test_idx) / len(df):.1%}"
    )

    print("\n[4/6] Train LightGBM")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_lgb)
    dtrain = lgb.Dataset(X_train_lgb, label=y_train_lgb, weight=sample_weights)
    dval = lgb.Dataset(X_val_lgb, label=y_val_lgb, reference=dtrain)

    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": SEED,
        "n_jobs": -1,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    save_lgb_model(model)
    print(f"  best_iteration={model.best_iteration}")
    print(f"  model={MODEL_PKL}")
    print(f"  feature_columns={FEATURE_COLUMNS_JSON}")

    print("\n[5/6] Test-set inference")
    proba = predict_in_batches(model, X_test)
    pred = np.argmax(proba, axis=1)
    test_raw_df = raw_df.iloc[test_idx].reset_index(drop=True)
    test_processed_df = df.iloc[test_idx].reset_index(drop=True)
    out_df = write_test_outputs(test_processed_df, y_test, proba)
    write_detection_assets(test_raw_df, out_df, proba)
    print(f"  results={OUTPUT_CSV}")
    print(f"  assets={ASSETS_DIR}")

    print("\n[6/6] Baseline comparison")
    if run_baselines:
        results_df, proba_by_model = evaluate_baselines(
            X_train_baseline,
            X_test,
            y_train_val,
            y_test,
            pred,
            proba,
        )
    else:
        results_df = pd.DataFrame(
            [metrics_row("LightGBM (ours)", y_test, pred)],
            columns=["Model", "Accuracy", "F1", "Precision", "Recall"],
        )
        proba_by_model = {"LightGBM (ours)": proba}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "model_comparison_with_baselines.csv"
    results_df.to_csv(report_path, index=False)
    write_weighted_metrics_plot(results_df)
    write_roc_report(y_test, proba_by_model)
    print(results_df.to_string(index=False))
    print(f"  report={report_path}")
    print(f"  confusion_matrix_png={REPORT_DIR / 'confusion_matrix.png'}")
    print(f"  weighted_metrics_png={REPORT_DIR / 'weighted_metrics_comparison.png'}")
    print(f"  roc={REPORT_DIR / 'roc_curve.png'}")

    return model, results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightGBM detector with semantic feature reduction")
    parser.add_argument("--reducer", choices=["supcon", "pca", "none"], default="supcon")
    parser.add_argument("--n-components", type=int, default=64, help="PCA target dimension")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    main(
        n_components=args.n_components,
        reducer_name=args.reducer,
        run_baselines=not args.skip_baselines,
    )
