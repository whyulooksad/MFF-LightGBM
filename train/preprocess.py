"""
步骤1：数据预处理
输入：任务一的 CSV 文件
输出：data/processed/flows.jsonl（每条流一行 JSON）

核心逻辑：
1. 按五元组归组（双向合并）
2. 组内按时间排序
3. 从 context_text 中提取 SSL/X509 事件，压缩成紧凑 JSON
4. 拼接为流文本序列
5. 从 summary_json 提取数值特征
6. 输出 JSONL
"""

import json
import re
import pandas as pd
from json.decoder import JSONDecoder
from collections import defaultdict
from tqdm import tqdm

from config import (
    EXAMPLE_CSV,
    INPUT_CSV,
    FLOWS_JSONL,
    SSL_FIELDS,
    X509_FIELDS,
    NUM_FEATURES,
)


# ============================================================
# 工具函数
# ============================================================

def canonical_flow_id(row):
    """
    计算与方向无关的流 ID。
    (A:pa → B:pb) 和 (B:pb → A:pa) 会得到同一个 ID。
    """
    ip_a, port_a = str(row["src_ip"]), int(row["src_port"])
    ip_b, port_b = str(row["dst_ip"]), int(row["dst_port"])
    proto = str(row["protocol"])

    if (ip_a, port_a) < (ip_b, port_b):
        return f"{ip_a}_{port_a}_{ip_b}_{port_b}_{proto}"
    else:
        return f"{ip_b}_{port_b}_{ip_a}_{port_a}_{proto}"


def parse_context_text(text: str) -> list[dict]:
    """
    从 context_text 单元格中解析出所有事件。

    context_text 格式（pandas 读取后，引号已转义回正常形式）：
        [SSL] {"ts": 123, "version": "TLSv12", ...}
        [X509] {"ts": 456, "certificate.subject": "CN=...", ...}

    一个单元格可能包含多个事件（用换行分隔），也可能只包含一个。

    返回: [{"type": "ssl", "data": {...}}, {"type": "x509", "data": {...}}, ...]
    """
    events = []
    decoder = JSONDecoder()
    pos = 0

    while pos < len(text):
        # 找到下一个 [SSL] 或 [X509] 标记
        ssl_pos = text.find("[SSL] ", pos)
        x509_pos = text.find("[X509] ", pos)

        if ssl_pos == -1 and x509_pos == -1:
            break

        # 取更靠前的那一个
        if ssl_pos == -1:
            start = x509_pos
            event_type = "x509"
            json_start = start + 7  # len("[X509] ")
        elif x509_pos == -1:
            start = ssl_pos
            event_type = "ssl"
            json_start = start + 6  # len("[SSL] ")
        elif ssl_pos < x509_pos:
            start = ssl_pos
            event_type = "ssl"
            json_start = start + 6
        else:
            start = x509_pos
            event_type = "x509"
            json_start = start + 7

        # 用 JSONDecoder.raw_decode 精确解析（自动处理嵌套大括号）
        try:
            obj, end_idx = decoder.raw_decode(text[json_start:])
            events.append({"type": event_type, "data": obj})
            pos = json_start + end_idx
        except json.JSONDecodeError:
            # 解析失败则跳过这个字符继续
            pos = json_start + 1

    return events


def compact_ssl_event(data: dict) -> dict:
    """将 SSL 事件压缩为只含关键字段的紧凑 dict。"""
    out = {"t": "s"}  # type 缩写
    for src_key, dst_key in SSL_FIELDS.items():
        if src_key in data and data[src_key] is not None:
            val = data[src_key]
            if isinstance(val, bool):
                val = 1 if val else 0
            elif isinstance(val, str):
                # 去掉字符串里可能混淆解析的字符
                val = val.replace('"', "'").replace("\n", " ").strip()
            out[dst_key] = val
    return out


def compact_x509_event(data: dict) -> dict:
    """将 X509 事件压缩为只含关键字段的紧凑 dict。"""
    out = {"t": "x"}
    for src_key, dst_key in X509_FIELDS.items():
        if src_key in data and data[src_key] is not None:
            val = data[src_key]
            if isinstance(val, str):
                val = val.replace('"', "'").replace("\n", " ").strip()
            out[dst_key] = val
    return out


def compact_event(event: dict) -> str:
    """将一个事件（ssl 或 x509）转成紧凑的 JSON 字符串。"""
    if event["type"] == "ssl":
        compact = compact_ssl_event(event["data"])
    else:
        compact = compact_x509_event(event["data"])
    # separators 去掉冒号和逗号后的空格，最紧凑
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)


def build_flow_text(events: list[dict]) -> str:
    """把一条流的所有事件拼接成文本。[CLS]/[SEP] 由 tokenizer 自动加。"""
    parts = []
    for ev in events:
        parts.append(compact_event(ev))
    return " ".join(parts)


def extract_num_features(summary_json_str) -> dict:
    """从 summary_json 字符串中提取数值特征。"""
    features = {}
    if pd.isna(summary_json_str) or not summary_json_str:
        return features

    try:
        data = json.loads(summary_json_str)
    except (json.JSONDecodeError, TypeError):
        return features

    for path_parts, col_name in NUM_FEATURES:
        # 沿着路径深入到嵌套 JSON 中取值
        val = data
        for key in path_parts:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
        # 处理取值
        if val is None:
            features[col_name] = None
        elif isinstance(val, bool):
            features[col_name] = 1 if val else 0
        elif isinstance(val, (int, float)):
            features[col_name] = val
        else:
            features[col_name] = None
    return features


# ============================================================
# 主流程
# ============================================================

def preprocess(csv_path: str = None, output_path: str = None, nrows: int = None):
    """
    主预处理函数。

    参数:
        csv_path: 输入 CSV 路径，默认用样例数据
        output_path: 输出 JSONL 路径，默认用 config 中的路径
        nrows: 只读前 nrows 行（测试用，None=全读）
    """
    if csv_path is None:
        # 优先用正式数据，没有则用样例
        csv_path = INPUT_CSV
        import os
        if not os.path.exists(csv_path):
            print(f"[INFO] 正式数据不存在 ({csv_path})，使用样例数据")
            csv_path = EXAMPLE_CSV

    if output_path is None:
        output_path = FLOWS_JSONL

    print(f"[1/5] 读取 CSV: {csv_path}")
    if nrows:
        print(f"      (测试模式，只读前 {nrows:,} 行)")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"      行数: {len(df):,}")
    print(f"      列: {list(df.columns)}")

    # ========================================
    # 计算 flow_id
    # ========================================
    print("[2/5] 计算 flow_id（双向合并）...")
    df["flow_id"] = df.apply(canonical_flow_id, axis=1)
    unique_flows = df["flow_id"].nunique()
    print(f"      唯一流数: {unique_flows:,}")

    # ========================================
    # 按 flow_id 分组处理
    # ========================================
    print("[3/5] 按流归组、提取事件...")

    flows_output = []
    # 统计
    total_events = 0
    total_ssl = 0
    total_x509 = 0

    grouped = df.groupby("flow_id")

    for fid, group in tqdm(grouped, desc="处理流", unit="流"):
        # 按时间排序
        group = group.sort_values("timestamp")

        # 从 context_text 提取所有事件
        all_events = []
        for _, row in group.iterrows():
            ctx = row["context_text"]
            if pd.notna(ctx):
                all_events.extend(parse_context_text(str(ctx)))

        # 统计事件类型
        total_events += len(all_events)
        total_ssl += sum(1 for e in all_events if e["type"] == "ssl")
        total_x509 += sum(1 for e in all_events if e["type"] == "x509")

        # 取 label（同一条流的所有行 label 应该一致，取第一个非空）
        label = None
        for _, row in group.iterrows():
            if pd.notna(row["label"]):
                label = int(row["label"])
                break

        # 取 summary_json（取第一个非空的）
        summary_json_str = None
        for _, row in group.iterrows():
            if pd.notna(row["summary_json"]):
                summary_json_str = str(row["summary_json"])
                break

        # 构建流文本
        text = build_flow_text(all_events)

        # 提取数值特征
        num_features = extract_num_features(summary_json_str)

        # 确定代表性的 src_ip / dst_ip（用于后续追溯）
        first_row = group.iloc[0]
        src_ip = str(first_row["src_ip"])
        dst_ip = str(first_row["dst_ip"])

        flows_output.append({
            "flow_id": fid,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "text": text,
            "label": label,
            "num_events": len(all_events),
            "num_features": num_features,
        })

    # ========================================
    # 统计信息
    # ========================================
    print(f"\n[4/5] 统计信息:")
    print(f"      总流数:      {len(flows_output):,}")
    print(f"      总事件数:     {total_events:,}")
    print(f"        SSL 事件:   {total_ssl:,}")
    print(f"        X509 事件:  {total_x509:,}")

    label_counts = defaultdict(int)
    for f in flows_output:
        label_counts[f["label"]] += 1
    print(f"      label 分布:  {dict(label_counts)}")

    # 事件数分布
    event_counts = [f["num_events"] for f in flows_output]
    print(f"      每条流事件数: min={min(event_counts)}, max={max(event_counts)}, "
          f"avg={sum(event_counts)/len(event_counts):.1f}")

    # ========================================
    # 保存 JSONL
    # ========================================
    print(f"\n[5/5] 保存到: {output_path}")
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in flows_output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"      输出 {len(flows_output):,} 条流")
    print("预处理完成!")
    return flows_output


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 测试：只取前 5000 行验证流程
    # 正式跑：preprocess() 不加 nrows
    preprocess(nrows=5000)
