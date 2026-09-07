"""Load the released LightGBM artefacts and perform batched prediction.
    一条网络流
    │
    ├── 80个人工网络特征
    │
    └── 流量文本
            ↓
        DeBERTa
            ↓
        768维语义特征
            ↓
        SupCon-AE
            ↓
        64维语义特征
            │
            └──────────────┐
                            ▼
                    64维语义 + 80维人工
                            │
                            ▼
                LightGBMDetector.prepare()
                ├──检查所需列
                ├──按照训练顺序排列
                ├──转换为数字
                ├──用训练中位数填缺失
                └──拒绝未覆盖的缺失值
                            │
                            ▼
                LightGBM多棵决策树
                            │
                            ▼
            [正常概率, 广告软件概率, DNS隧道概率,
            勒索软件概率, ……]
                            │
                ┌──────────┴──────────┐
                ▼                     ▼
        argmax找到最大概率位置     max取得最大概率
                │                     │
                ▼                     ▼
            最终类别                置信度

"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class LightGBMDetector:
    def __init__(self, detector_dir: str | Path, batch_size: int = 4096):
        root = Path(detector_dir)
        self.batch_size = batch_size
        self.feature_columns = json.loads((root / "feature_columns.json").read_text(encoding="utf-8"))
        medians_path = root / "imputation_medians.json"
        self.medians = json.loads(medians_path.read_text(encoding="utf-8")) if medians_path.exists() else {}
        with (root / "best_lgb_model.pkl").open("rb") as stream:
            self.model = pickle.load(stream)

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"推理特征缺少训练时的列: {missing[:5]} ...")
        values = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        for column, median in self.medians.items():
            if column in values:
                values[column] = values[column].fillna(median)
        remaining = [column for column in values.columns if values[column].isna().any()]
        if remaining:
            raise ValueError(
                "推理出现训练契约未覆盖的缺失值，拒绝使用当前上传批次统计量填充: "
                + ", ".join(remaining[:8])
            )
        return values

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.prepare(frame)
        best_iteration = getattr(self.model, "best_iteration", None)
        chunks = []
        for start in range(0, len(values), self.batch_size):
            batch = values.iloc[start:start + self.batch_size].to_numpy(dtype=np.float32)
            chunks.append(self.model.predict(batch, num_iteration=best_iteration))
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(chunks)
