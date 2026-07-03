# -*- coding: utf-8 -*-
"""Feature ablation for LightGBM detector.

Runs the same LightGBM train/val/test split on three feature groups:
1. fused: LLM feat_* + handcrafted numeric features
2. llm_only: only LLM feat_* features
3. manual_only: only handcrafted numeric flow features
"""

from __future__ import annotations

import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from LLM_train.config import ID2LABEL
from pipeline.config import FEATURES_FUSED_CSV, PIPELINE_OUTPUT_DIR, ROOT
from pipeline.detector import (
    LABEL_COL,
    SEED,
    TEST_SIZE,
    VAL_SIZE_WITHIN_TRAIN,
    detector_feature_columns,
    preprocess_detector_dataframe,
)


REPORT_DIR = PIPELINE_OUTPUT_DIR / "detector_report" / "feature_ablation"
CHECKPOINT_DIR = ROOT / "checkpoints" / "detector" / "feature_ablation"
BATCH_SIZE = 4096

METRIC_PALETTE = {
    "Accuracy": "#FFB347",
    "Precision": "#4682B4",
    "Recall": "#ADD8E6",
    "Weighted F1": "#C4A0E0",
}


def lgb_params(num_classes: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
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


def split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        indices,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        random_state=SEED,
        stratify=y_train_val,
    )
    return train_idx, val_idx, test_idx, y_train, y_val


def predict_in_batches(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    chunks = []
    best_iteration = getattr(model, "best_iteration", None)
    for start in range(0, len(X), BATCH_SIZE):
        batch = X[start : start + BATCH_SIZE]
        if best_iteration:
            chunks.append(model.predict(batch, num_iteration=best_iteration))
        else:
            chunks.append(model.predict(batch))
    return np.vstack(chunks)


def metrics_row(group: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Feature Group": group,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
    }


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    all_features = detector_feature_columns(df)
    llm_features = [col for col in all_features if col.startswith("feat_")]
    manual_features = [col for col in all_features if not col.startswith("feat_")]
    return {
        "Fused": all_features,
        "LLM-only": llm_features,
        "Manual-only": manual_features,
    }


def train_one_group(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict, np.ndarray]:
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    dtrain = lgb.Dataset(
        X[train_idx],
        label=y_train,
        weight=compute_sample_weight(class_weight="balanced", y=y_train),
    )
    dval = lgb.Dataset(X[val_idx], label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params(num_classes=len(np.unique(y))),
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
    )

    safe_name = name.lower().replace("-", "_")
    with open(CHECKPOINT_DIR / f"{safe_name}_lgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    proba = predict_in_batches(model, X[test_idx])
    pred = np.argmax(proba, axis=1)

    group_dir = REPORT_DIR / safe_name
    group_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_test,
        pred,
        labels=sorted(ID2LABEL),
        target_names=[ID2LABEL[idx] for idx in sorted(ID2LABEL)],
        digits=4,
        zero_division=0,
    )
    (group_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    return metrics_row(name, y_test, pred), proba


def plot_metrics(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df.melt(
        id_vars="Feature Group",
        value_vars=["Accuracy", "Precision", "Recall", "Weighted F1"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(10.5, 6.2))
    ax = sns.barplot(data=plot_df, x="Feature Group", y="Score", hue="Metric", palette=METRIC_PALETTE)
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature Ablation: Weighted Metrics Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_ablation_metrics.png", dpi=200)
    plt.close()


def plot_roc(y_test: np.ndarray, proba_by_group: dict[str, np.ndarray]) -> None:
    classes = np.array(sorted(ID2LABEL))
    y_bin = label_binarize(y_test, classes=classes)
    auc_rows = []

    plt.figure(figsize=(9, 6.5))
    for name, proba in proba_by_group.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        micro_auc = roc_auc_score(y_bin, proba, average="micro", multi_class="ovr")
        auc_rows.append({"Feature Group": name, "micro_auc": micro_auc})
        linewidth = 2.8 if name == "Fused" else 1.9
        linestyle = "--" if name == "Fused" else "-"
        plt.plot(fpr, tpr, linewidth=linewidth, linestyle=linestyle, label=f"{name} micro AUC={micro_auc:.3f}")

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Feature Ablation: Micro-Average ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_ablation_roc.png", dpi=200)
    plt.close()
    pd.DataFrame(auc_rows).to_csv(REPORT_DIR / "feature_ablation_auc.csv", index=False)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" LightGBM feature ablation: fused / LLM-only / manual-only")
    print("=" * 60)
    print(f"[1/4] Read fused features: {FEATURES_FUSED_CSV}")
    raw_df = pd.read_csv(FEATURES_FUSED_CSV)
    df = preprocess_detector_dataframe(raw_df)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Feature ablation requires a '{LABEL_COL}' column")

    y = df[LABEL_COL].astype("int64").to_numpy()
    train_idx, val_idx, test_idx, _, _ = split_indices(y)
    y_test = y[test_idx]

    metrics_rows = []
    proba_by_group = {}
    groups = feature_groups(df)

    print("[2/4] Train and test each feature group")
    for name, cols in groups.items():
        if not cols:
            raise ValueError(f"No features found for group: {name}")
        print(f"  {name}: features={len(cols)}")
        X = df[cols].values.astype(np.float32)
        row, proba = train_one_group(name, X, y, train_idx, val_idx, test_idx)
        metrics_rows.append(row)
        proba_by_group[name] = proba

    print("[3/4] Write metrics and plots")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(REPORT_DIR / "feature_ablation_metrics.csv", index=False)
    plot_metrics(metrics_df)
    plot_roc(y_test, proba_by_group)

    print("[4/4] Done")
    print(metrics_df.to_string(index=False))
    print(f"  report_dir={REPORT_DIR}")


if __name__ == "__main__":
    main()
