"""Runtime-safe flow JSONL input helpers shared by development and product code."""

from __future__ import annotations

import json
from pathlib import Path


def load_flows(jsonl_path: str | Path) -> list[dict]:
    """Load serialized flows without importing any training module."""
    path = Path(jsonl_path)
    flows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    labels: dict[object, int] = {}
    for flow in flows:
        label = flow.get("label")
        labels[label] = labels.get(label, 0) + 1

    print(f"加载 {len(flows):,} 条流 from {path}")
    print(f"  label 分布: {labels}")
    return flows
