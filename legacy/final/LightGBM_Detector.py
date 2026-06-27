# -*- coding: utf-8 -*-
"""
detect_malware.py  (完整版：包含前端可视化数据导出)
==================================================
基于已训练的分类器对降维后的流量特征进行恶意加密流量检测，
并生成一系列可直接用于前端绘图的 CSV/JSON 文件。

用法：直接运行本脚本，修改顶部的路径配置即可。
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ====================== 可调配置 (直接修改此处) ======================
# 输入降维后的 CSV（由 supcon_ae.py 或特征工程生成）
INPUT_CSV = "reduced_features.csv"
# 输出检测结果 CSV（包含每条流的详细预测）
OUTPUT_CSV = "detection_results.csv"
# 前端可视化用数据文件保存目录
OUTPUT_DIR = "detection_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型文件路径
MODEL_DIR = "./models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_FILES = {
    "RandomForest": os.path.join(MODEL_DIR, "rf_model.pkl"),
    "ExtraTrees":   os.path.join(MODEL_DIR, "et_model.pkl"),
    "LightGBM":     os.path.join(MODEL_DIR, "lgb_model.txt"),
    "XGBoost":      os.path.join(MODEL_DIR, "xgb_model.json"),
}
ACTIVE_MODELS = ["RandomForest", "ExtraTrees", "LightGBM"]  # 可修改

# 标识列（保留至输出文件，不参与预测）
ID_COLS = ["flow_uid"]
# 特征列前缀（降维特征通常以 z_ 开头）
FEATURE_PREFIX = "z_"
# 高置信恶意流阈值（大于该值展示在恶意详情中）
MALICIOUS_CONF_THRESHOLD = 0.8
# 批预测大小
BATCH_SIZE = 4096
# 趋势分析：若有时间戳则用列名，否则按样本序号分段
TIME_COL = "timestamp"          # 如果存在，将用于趋势计算；不存在则自动忽略
TREND_WINDOW = 100              # 每批样本数（时间窗方式时无效）
# 降维可视化是否使用 PCA 降到 2D（使用已有的 z_ 特征）
USE_PCA_2D = True
RANDOM_STATE = 42

# ====================== 工具函数 ======================

def load_models(model_dir, model_files, active_models):
    loaded = {}
    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"标准化器已加载: {SCALER_PATH}")
    else:
        print("警告：未找到标准化器文件，将不对特征进行标准化。")

    for name, path in model_files.items():
        if active_models and name not in active_models:
            continue
        if not os.path.exists(path):
            print(f"警告：模型文件不存在: {path}，跳过 {name}")
            continue
        if name == "LightGBM":
            import lightgbm as lgb
            try:
                model = lgb.Booster(model_file=path)
            except:
                model = joblib.load(path)
        elif name == "XGBoost":
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(path)
        else:
            model = joblib.load(path)
        loaded[name] = model
        print(f"模型已加载: {name} from {path}")
    return loaded, scaler

def predict_batch(model_dict, X):
    """
    返回字典，包含每个模型的预测标签、置信度（最大概率）以及平均投票结果。
    """
    n_samples = X.shape[0]
    probs_dict = {}
    for name, model in model_dict.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        elif name == "LightGBM":
            # LightGBM Booster 的预测概率
            proba = model.predict(X, raw_score=False)
            if proba.ndim == 1:
                # 尝试获取类别概率（可能需要 pred_leaf 等）
                proba = model.predict(X, raw_score=False, pred_leaf=False)
        elif name == "XGBoost":
            import xgboost as xgb
            dtest = xgb.DMatrix(X)
            proba = model.predict(dtest)
        else:
            raise ValueError(f"不支持的模型类型: {name}")
        # 确保概率为二维 (n_samples, n_classes)
        if proba.ndim == 1:
            # 如果是单列，可能是回归或二分类，扩展为两列
            if np.max(proba) > 1:  # 简单的启发式
                pass
            else:
                proba = np.vstack([1 - proba, proba]).T
        probs_dict[name] = proba

    # 构建输出
    results = {}
    for name, proba in probs_dict.items():
        pred = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
        results[f"pred_{name}"] = pred
        results[f"conf_{name}"] = conf

    if len(model_dict) > 1:
        all_preds = [results[f"pred_{name}"] for name in model_dict]
        vote_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0,
                                        arr=np.array(all_preds))
        results["vote_pred"] = vote_pred
        avg_proba = np.mean(list(probs_dict.values()), axis=0)
        results["avg_conf"] = np.max(avg_proba, axis=1)

    return results, probs_dict

# ====================== 前端数据生成 ======================

def generate_frontend_assets(df, results, probs_dict, output_dir):
    """
    根据检测结果生成前端可视化所需的数据文件。
    """
    print("\n生成前端可视化数据...")

    # 1. 统计概览 (detection_summary.csv)
    total = len(df)
    vote_pred = results.get("vote_pred", results[list(results.keys())[0]])  # 默认用投票或第一个模型
    # 统计各类别数量
    class_counts = pd.Series(vote_pred).value_counts().to_dict()
    # 假设类别 0 为良性，其他为恶意（可根据实际情况自定义）
    benign_label = 0
    malicious_count = sum(v for k, v in class_counts.items() if k != benign_label)
    summary = {
        "total_flows": total,
        "predicted_benign": class_counts.get(benign_label, 0),
        "predicted_malicious": malicious_count,
    }
    # 详细类别计数（可列出所有类别）
    for cls, cnt in class_counts.items():
        summary[f"class_{cls}_count"] = cnt
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "detection_summary.csv"), index=False)
    print("  - detection_summary.csv: 检测概览统计")

    # 2. 风险分布 (每样本置信度)
    risk_cols = [col for col in results if col.startswith("conf_") or col == "avg_conf"]
    risk_df = df[ID_COLS].copy() if ID_COLS else pd.DataFrame(index=df.index)
    for col in risk_cols:
        risk_df[col] = results[col]
    risk_df.to_csv(os.path.join(output_dir, "risk_distribution.csv"), index=False)
    print("  - risk_distribution.csv: 每个流的置信度，用于直方图/箱线图")

    # 3. 恶意流详情 (高置信恶意流)
    # 使用平均置信度或投票结果
    conf_key = "avg_conf" if "avg_conf" in results else next((c for c in results if c.startswith("conf_")), None)
    is_mal = vote_pred != benign_label  # 预测为恶意
    high_conf_mask = (results.get(conf_key, 0) > MALICIOUS_CONF_THRESHOLD)
    malicious_idx = np.where(is_mal & high_conf_mask)[0]
    if len(malicious_idx) > 0:
        mal_df = df.iloc[malicious_idx].copy()
        # 添加预测信息
        for col in results:
            if col.startswith("pred_") or col.startswith("conf_"):
                # 确保对应的是标量而非数组
                arr = results[col]
                if isinstance(arr, np.ndarray):
                    mal_df[col] = arr[malicious_idx]
                elif isinstance(arr, list):
                    mal_df[col] = np.array(arr)[malicious_idx]
        mal_df.to_csv(os.path.join(output_dir, "malicious_details.csv"), index=False)
    else:
        pd.DataFrame(columns=df.columns).to_csv(os.path.join(output_dir, "malicious_details.csv"), index=False)
    print("  - malicious_details.csv: 高置信恶意流详情，便于表格展示")

    # 4. 趋势分析
    if TIME_COL in df.columns and pd.api.types.is_numeric_dtype(df[TIME_COL]):
        time_series = df[TIME_COL].values
        # 按时间排序并划分窗口（例如每分钟或每10秒）
        sort_idx = np.argsort(time_series)
        sorted_df = df.iloc[sort_idx]
        sorted_time = time_series[sort_idx]
        sorted_pred = vote_pred[sort_idx] if isinstance(vote_pred, np.ndarray) else np.array(list(vote_pred))[sort_idx]
        # 按固定时间窗口统计（简单起分10个区间）
        bins = np.linspace(sorted_time.min(), sorted_time.max(), 11)
        windows = pd.cut(sorted_time, bins=bins, labels=False, right=False)
        trend_rows = []
        for win in range(10):
            mask = windows == win
            total = mask.sum()
            malicious = (sorted_pred[mask] != benign_label).sum() if total > 0 else 0
            trend_rows.append({
                "window_start": bins[win],
                "window_end": bins[win+1],
                "total": total,
                "malicious": malicious
            })
        trend_df = pd.DataFrame(trend_rows)
    else:
        # 没有时间戳，按样本序号分段
        total = len(df)
        pred_arr = vote_pred if isinstance(vote_pred, np.ndarray) else np.array(list(vote_pred))
        trend_rows = []
        for start in range(0, total, TREND_WINDOW):
            end = min(start + TREND_WINDOW, total)
            segment = pred_arr[start:end]
            trend_rows.append({
                "sample_start": start,
                "sample_end": end - 1,
                "total": len(segment),
                "malicious": int(np.sum(segment != benign_label))
            })
        trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(os.path.join(output_dir, "trend_data.csv"), index=False)
    print("  - trend_data.csv: 按时间或序号分段的恶意流趋势")

    # 5. 模型投票一致性
    pred_cols = [col for col in results if col.startswith("pred_") and "vote" not in col]
    if len(pred_cols) > 1:
        vote_df = pd.DataFrame()
        if ID_COLS:
            vote_df[ID_COLS] = df[ID_COLS]
        for col in pred_cols:
            vote_df[col] = results[col]
        vote_df.to_csv(os.path.join(output_dir, "vote_consistency.csv"), index=False)
        print("  - vote_consistency.csv: 各模型预测标签，用于投票一致性分析")
    else:
        print("  - 跳过多模型一致性（仅有一个模型）")

    # 6. 降维可视化 (2D)
    if USE_PCA_2D:
        z_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
        if z_cols:
            X_z = df[z_cols].values.astype(np.float32)
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            ld2 = pca.fit_transform(X_z)
            pca_df = df[ID_COLS].copy() if ID_COLS else pd.DataFrame(index=df.index)
            pca_df["x"] = ld2[:, 0]
            pca_df["y"] = ld2[:, 1]
            # 添加投票预测标签
            pca_df["pred_label"] = vote_pred
            pca_df.to_csv(os.path.join(output_dir, "latent_2d.csv"), index=False)
            print("  - latent_2d.csv: 2D 降维坐标及预测标签，可用于散点图")
        else:
            print("  - 未找到 z_ 特征列，跳过 2D 降维")

    print("所有前端数据文件已生成至:", output_dir)

# ====================== 主程序 ======================

def main():
    print("=" * 60)
    print(" 恶意加密流量检测（含前端数据导出）")
    print("=" * 60)

    # 1. 读取数据
    print(f"\n[1/3] 读取输入 CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  原始形状: {df.shape}")

    # 提取特征列
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise ValueError(f"未找到以 '{FEATURE_PREFIX}' 开头的特征列。")
    print(f"  检测到的特征列: {len(feature_cols)} 维")

    # 确保标识列存在
    id_cols_exist = [c for c in ID_COLS if c in df.columns]
    if not id_cols_exist:
        df["flow_uid"] = df.index.astype(str)
        id_cols_exist = ["flow_uid"]

    X = df[feature_cols].values.astype(np.float32)

    # 2. 加载模型
    print(f"\n[2/3] 加载模型...")
    models, scaler = load_models(MODEL_DIR, MODEL_FILES, ACTIVE_MODELS)
    if not models:
        print("错误：没有成功加载任何模型，退出。")
        return

    if scaler is not None:
        X = scaler.transform(X)

    # 3. 推理
    print(f"\n[3/3] 执行推理...")
    all_results = {}
    all_probs = None   # 暂存概率，如果只需要结果可以不存全部概率，但为了前端我们可能不需要，这里略
    start_idx = 0
    while start_idx < len(X):
        end_idx = min(start_idx + BATCH_SIZE, len(X))
        X_batch = X[start_idx:end_idx]
        batch_res, batch_probs = predict_batch(models, X_batch)
        for key, val in batch_res.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].extend(val if isinstance(val, (list, np.ndarray)) else [val])
        start_idx = end_idx

    # 组装最终检测结果 DataFrame
    out_df = df[id_cols_exist].copy()
    for col_name, col_data in all_results.items():
        out_df[col_name] = col_data

    # 保存原始检测结果
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"检测结果已保存至: {OUTPUT_CSV}")

    # 生成前端可视化数据
    # 注意：为了生成这些数据，我们需要 results 字典（全局）和 probs 字典（可选），
    # 函数 generate_frontend_assets 内部会使用 all_results 和 df。
    # 传递精简的概率数据可能不需要，我们可忽略 probs 参数。
    generate_frontend_assets(df, all_results, None, OUTPUT_DIR)

    print("\n全部任务完成！")

if __name__ == "__main__":
    main()