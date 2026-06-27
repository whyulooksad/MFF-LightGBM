# -*- coding: utf-8 -*-
"""
multi_classifier_eval.py
========================
多分类器对比评估脚本。
针对降维后的列结构，仅自动删除 flow_uid 和 label_name 两列。
请确保输入 CSV 中已提前移除所有可能导致信息泄露的列（如 IP、端口等）。

运行： python multi_classifier_eval.py
"""

import os, re, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("XGBoost 未安装，将跳过该模型。如需安装：pip install xgboost")

warnings.filterwarnings('ignore')

# ======================== 配置 ========================
INPUT_CSV = "D:/jinxian/Pycharm/比赛/data/output/features_fused.csv"
LABEL_COL = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.3
VALID_SIZE = 0.1

# 仅删除这两个标识列，不再删除其他列（假设原始数据已清理）
DROP_COLS = ["flow_uid", "label_name"]

# 分类器参数
RF_PARAMS = {
    'n_estimators': 200,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}
ET_PARAMS = {
    'n_estimators': 200,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}
XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'verbosity': 0
}
LGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1
}

# ======================== 预处理 ========================

def extract_cn_features(df):
    """将 cn_value 转换为数值特征"""
    if 'cn_value' not in df.columns:
        return pd.DataFrame(index=df.index)
    cn = df['cn_value'].fillna('').astype(str)
    feats = pd.DataFrame(index=df.index)
    feats['cn_len'] = cn.str.len()
    tld = cn.str.split('.').str[-1].str.lower()
    tld_counts = tld.value_counts()
    common_tlds = tld_counts[tld_counts >= 5].index.tolist()
    tld = tld.apply(lambda x: x if x in common_tlds else 'other')
    feats['cn_tld_encoded'] = LabelEncoder().fit_transform(tld)
    def count_subdomains(domain):
        parts = domain.split('.')
        if parts and parts[-1] == '': parts = parts[:-1]
        return max(0, len(parts) - 1)
    feats['cn_subdomain_count'] = cn.apply(count_subdomains)
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    feats['cn_has_ip'] = cn.apply(lambda x: 1 if ip_pattern.match(x) else 0)
    def calc_entropy(s):
        if len(s) == 0: return 0.0
        counts = Counter(s)
        probs = np.array(list(counts.values())) / len(s)
        return entropy(probs, base=2)
    feats['cn_entropy'] = cn.apply(calc_entropy)
    feats['cn_is_www'] = cn.str.startswith('www.').astype(int)
    feats['cn_digit_ratio_new'] = cn.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    feats['cn_has_hyphen'] = cn.str.contains('-').astype(int)
    return feats


def preprocess_dataframe(df):
    """
    清洗 DataFrame，返回只含数值特征和标签的 DataFrame。
    1. 删除标识列（flow_uid, label_name），保留 label 作为标签。
    2. 若存在 'cn_value'，提取域名统计特征并删除原列。
    3. 对非数值列进行编码或删除。
    4. 填充缺失值。
    """
    protected = {LABEL_COL}  # 只保护标签列

    # 删除指定的标识列
    existing_drop = [c for c in DROP_COLS if c in df.columns and c not in protected]
    if existing_drop:
        print(f"\n删除了 {len(existing_drop)} 个标识列：{existing_drop}")
        df.drop(columns=existing_drop, inplace=True)
    else:
        print("未发现需删除的标识列。")

    # cn_value 特征提取
    cn_feats = extract_cn_features(df)
    if not cn_feats.empty:
        df = pd.concat([df, cn_feats], axis=1)
        if 'cn_value' in df.columns:
            df.drop(columns=['cn_value'], inplace=True)

    # 处理其他非数值列（排除标签）
    non_num_cols = df.select_dtypes(include=['object']).columns.tolist()
    non_num_cols = [c for c in non_num_cols if c != LABEL_COL]
    for col in non_num_cols:
        if df[col].nunique() < 50:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df.drop(columns=[col], inplace=True)

    # 缺失值填充
    num_cols = df.select_dtypes(include=[np.number]).columns
    if df[num_cols].isnull().any().any():
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


# ======================== 评估与绘图 ========================

def evaluate_model(name, model, X_train, y_train, X_test, y_test, classes, is_lgb=False):
    """训练并预测，返回预测标签、预测概率、宏F1、混淆矩阵等"""
    if is_lgb:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(20)])
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = model.predict_proba(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    # 确保 target_names 为字符串列表
    str_classes = [str(c) for c in classes]
    report = classification_report(y_test, y_pred, target_names=str_classes, digits=4)
    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'macro_f1': macro_f1,
        'cm': cm,
        'report': report,
        'str_classes': str_classes
    }


def plot_confusion_matrices(results_dict, save_path="conf_matrices_compare.png"):
    """绘制所有分类器的混淆矩阵（行归一化）"""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results_dict.items()):
        cm = res['cm']
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        labels = res.get('str_classes', range(len(cm)))
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_title(f"{name} (Macro F1={res['macro_f1']:.3f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(results_dict, y_test, classes, save_path="roc_curves_compare.png"):
    """绘制所有分类器的微平均 ROC 曲线对比图"""
    from sklearn.preprocessing import label_binarize
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("husl", len(results_dict))
    for (name, res), color in zip(results_dict.items(), colors):
        y_proba = res['y_proba']
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro‑average ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_f1_comparison(results_dict, save_path="f1_comparison.png"):
    """柱状图对比宏F1"""
    names = list(results_dict.keys())
    f1s = [results_dict[n]['macro_f1'] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, f1s, color=sns.color_palette("Set2"))
    for bar, score in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{score:.4f}", ha='center', va='bottom')
    plt.ylim(0, 1.05)
    plt.ylabel("Macro F1 Score")
    plt.title("Classifier Macro F1 Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ======================== 主程序 ========================

def main():
    print("=" * 60)
    print(" 多分类器对比评估（LightGBM / RandomForest / XGBoost / ExtraTrees）")
    print("=" * 60)

    # 1. 加载并预处理
    df = pd.read_csv(INPUT_CSV)
    print(f"原始形状: {df.shape}")
    df = preprocess_dataframe(df)

    # 分离特征和标签
    feature_cols = [c for c in df.columns
                    if c != LABEL_COL and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[LABEL_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)

    print(f"样本数: {X.shape[0]}, 特征维度: {X.shape[1]}, 类别数: {n_classes}")

    # 2. 分层划分训练/验证/测试集
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_ratio = VALID_SIZE / (1 - TEST_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trval, y_trval, test_size=val_ratio, stratify=y_trval, random_state=RANDOM_STATE
    )
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_valid.shape[0]}, 测试集: {X_test.shape[0]}")

    # 3. 定义所有分类器
    classifiers = {
        "LightGBM": lgb.LGBMClassifier(**LGB_PARAMS),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
        "ExtraTrees": ExtraTreesClassifier(**ET_PARAMS),
    }
    if _XGB_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(**XGB_PARAMS)

    results = {}
    for name, model in classifiers.items():
        print(f"\n--- 训练 {name} ---")
        is_lgb = isinstance(model, lgb.LGBMClassifier)
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test,
                             classes, is_lgb=is_lgb)
        results[name] = res
        print(f"Macro F1: {res['macro_f1']:.4f}")
        print(res['report'])

    # 4. 绘制可视化
    print("\n生成对比图表...")
    plot_confusion_matrices(results, "conf_matrices_compare.png")
    # 注意：绘制 ROC 时传入 classes 原始数组（可能为整数），但内部会处理为字符串，不影响绘图
    plot_roc_curves(results, y_test, classes, "roc_curves_compare.png")
    plot_f1_comparison(results, "f1_comparison.png")

    # 5. 保存详细报告
    with open("classification_summary.txt", "w", encoding="utf-8") as f:
        for name, res in results.items():
            f.write(f"=== {name} ===\nMacro F1: {res['macro_f1']:.4f}\n")
            f.write(res['report'] + "\n\n")
    print("详细报告已保存至 classification_summary.txt")
    print("完成。")


if __name__ == "__main__":
    main()