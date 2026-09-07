"""One authoritative, leak-safe split contract for all training stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_SPLITS = ("train", "validation", "test")


def predefined_split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in ("group_id", "split") if name not in frame.columns]
    if missing:
        raise ValueError(
            f"数据缺少 {missing}；禁止按 flow 随机切分。请先用 split_dataset.py 按 PCAP/sample/capture 分组"
        )
    if frame["group_id"].isna().any() or frame["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("group_id 存在空值；每条流必须能追溯到原始 PCAP/sample/capture")
    invalid = sorted(set(frame["split"].astype(str)) - set(VALID_SPLITS))
    if invalid:
        raise ValueError(f"非法 split 值: {invalid}; 只允许 {VALID_SPLITS}")
    ownership = frame.groupby("group_id")["split"].nunique()
    leaking = ownership[ownership > 1]
    if not leaking.empty:
        raise ValueError(f"发现跨集合的数据组（数据泄漏）: {leaking.index[:10].tolist()}")
    result = tuple(np.flatnonzero(frame["split"].to_numpy() == name) for name in VALID_SPLITS)
    if any(len(values) == 0 for values in result):
        raise ValueError(f"train/validation/test 都必须非空，实际数量: {[len(v) for v in result]}")
    if "label" in frame.columns:
        labels = frame["label"].astype(str)
        missing_pairs = [
            (label, split)
            for label in sorted(labels.unique())
            for split in VALID_SPLITS
            if not ((labels == label) & (frame["split"].astype(str) == split)).any()
        ]
        if missing_pairs:
            raise ValueError(f"部分标签没有覆盖全部集合: {missing_pairs[:12]}")
    return result


def split_flow_records(flows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    frame = pd.DataFrame(
        {
            "group_id": [flow.get("group_id") for flow in flows],
            "split": [flow.get("split") for flow in flows],
            "label": [flow.get("label") for flow in flows],
        }
    )
    indices = predefined_split_indices(frame)
    return tuple([[flows[int(index)] for index in part] for part in indices])
