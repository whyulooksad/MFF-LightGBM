"""Compare metrics.json files produced for multiple model releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"release_id": path.parent.name, **{key: metrics.get(key) for key in ("accuracy", "macro_f1", "weighted_f1")}})
    return sorted(rows, key=lambda row: (row.get("macro_f1") is not None, row.get("macro_f1") or 0), reverse=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.metrics), ensure_ascii=False, indent=2))
