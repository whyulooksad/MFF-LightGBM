import warnings

import pandas as pd

# 在导入任何其他库之前，彻底屏蔽所有烦人的 Pandas 警告
warnings.filterwarnings("ignore")

import os
import sys
import gc
import re
import traceback
from datetime import datetime
import time as time_lib
from tqdm import tqdm

# 开启 C 级别崩溃信号拦截器
import faulthandler

faulthandler.enable()

# 全局关闭 tqdm 内部监控线程
tqdm.monitor_interval = 0

# 原始 CSV 的存放根目录（请根据实际物理路径修改）
CSV_DIR_2017 = r"I:\CICMalAnal2017\csv"
CSV_DIR_2020 = r"I:\CIC-DoHBrw-2020\CSVs\CSV"

# 统一输出的清洗后 CSV 根目录（自动在您脚本目录下创建）
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV_DIR = os.path.join(WORK_DIR, "processed_csv")

# 物理基准：Unix 纪元起点，用于纯 Python 的时区无关高精转换
EPOCH_START = datetime(1970, 1, 1)

# 时区物理时差
OFFSET_2020 = -10800  # 2020 加拿大夏令时 (UTC-3)
OFFSET_2017 = 28800  # 2017 北京时间 (UTC+8)


# ==================== 【自适应列名映射辅助函数】 ====================

def get_adaptive_column_map(df_cols):
    col_map = {}
    for col in df_cols:
        # 标准化去噪：移除非字母数字字符（如空格、点、下划线），使 Source.IP、Source IP、SourceIP 统一化
        c_clean = re.sub(r'[^a-z0-9]', '', col.strip().lower())

        if "sourceip" in c_clean or "srcip" in c_clean:
            col_map["src_ip"] = col
        elif "destinationip" in c_clean or "dstip" in c_clean or "destip" in c_clean:
            col_map["dst_ip"] = col
        elif "sourceport" in c_clean or "srcport" in c_clean:
            col_map["src_port"] = col
        elif "destinationport" in c_clean or "dstport" in c_clean or "destport" in c_clean:
            col_map["dst_port"] = col
        elif c_clean == "timestamp" or c_clean == "time":
            col_map["timestamp"] = col
    return col_map


def clean_col_name(col):
    return re.sub(r'[^a-z0-9]', '', col.strip().lower())


# ==================== 双轨纯 Python 时区无关解析引擎 ====================

def parse_time_2017_to_epoch(ts_str):
    """
    2017 专属解析：支持 '%d/%m/%Y %H:%M:%S' 和带 AM/PM 的格式。
    """
    val = ts_str.strip().replace("下午", "PM").replace("上午", "AM")
    try:
        if "PM" in val or "AM" in val:
            dt = datetime.strptime(val, "%d/%m/%Y %I:%M:%S %p")
        else:
            dt = datetime.strptime(val, "%d/%m/%Y %H:%M:%S")
        epoch = int((dt - EPOCH_START).total_seconds())
        return epoch - OFFSET_2017
    except:
        return 0


def parse_time_2020_to_epoch(ts_str):
    """
    2020 专属解析：支持 '%Y/%m/%d %H:%M:%S' 和无秒数格式 '%Y/%m/%d %H:%M'。
    """
    val = ts_str.strip()
    try:
        dt = datetime.strptime(val, "%Y/%m/%d %H:%M:%S")
    except:
        try:
            dt = datetime.strptime(val, "%Y/%m/%d %H:%M")
        except:
            return 0
    epoch = int((dt - EPOCH_START).total_seconds())
    return epoch - OFFSET_2020


# ==================== 统一流式清洗主逻辑 ====================

def preprocess_all_csvs():
    print(f"正在建立清洗输出主文件夹: {PROCESSED_CSV_DIR}")
    os.makedirs(PROCESSED_CSV_DIR, exist_ok=True)

    # 1. 扫描两套数据集的所有任务
    tasks = []

    # 扫描 2017 嵌套目录
    if os.path.exists(CSV_DIR_2017):
        for root, _, files in os.walk(CSV_DIR_2017):
            for file in files:
                if file.endswith(".csv"):
                    # 采用全路径文本（目录名+文件名）不区分大小写匹配，兼顾扁平命名和子文件夹嵌套
                    full_lower_path = (root + os.sep + file).lower()
                    label = None
                    for candidate in ["adware", "smsmalware", "ransomware", "scareware", "benign"]:
                        if candidate in full_lower_path:
                            label = candidate
                            break
                    if label:
                        tasks.append((os.path.join(root, file), "CIC2017", label))

    # 扫描 2020 嵌套目录
    if os.path.exists(CSV_DIR_2020):
        for root, _, files in os.walk(CSV_DIR_2020):
            for file in files:
                # 支持 2020 任意扁平/嵌套文件名
                if file.endswith(".csv"):
                    full_lower_path = (root + os.sep + file).lower()
                    label = None
                    for candidate in ["dns2tcp", "dnscat2", "iodine", "benign"]:
                        if candidate in full_lower_path:
                            label = candidate
                            break
                    if label:
                        tasks.append((os.path.join(root, file), "DoH2020", label))

    if not tasks:
        print("未在指定的磁盘路径下发现任何原始 CSV 文件。请检查配置。")
        return

    print(f"\n共发现 {len(tasks)} 个合规原始日志主干文件。开始进行‘双轨时序对齐洗白’...")
    for csv_path, source, label in tqdm(tasks, desc="全局 CSV 清洗总进度"):
        filename = os.path.basename(csv_path)

        # 建立保存路径：D:\jinxian\Pycharm\比赛\processed_csv\<数据集>\<标签>\
        output_subdir = os.path.join(PROCESSED_CSV_DIR, source, label)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, filename.replace(".csv", "_clean.csv"))

        try:
            # 1. 表头读取（表头是纯英文字符，默认使用 utf-8 解析，由于无 data 读写，非常安全）
            # 【修复点】：彻底移除了 read_csv 中不支持的 errors 参数
            df_header = pd.read_csv(csv_path, nrows=0, encoding='utf-8')
            col_map = get_adaptive_column_map(df_header.columns)

            # 检查关键列是否全部被成功映射
            for col_key in ["src_ip", "src_port", "dst_ip", "dst_port", "timestamp"]:
                if col_key not in col_map:
                    raise KeyError(f"无法自适应匹配到关键列名: {col_key}")

            needed_cols = [col_map["src_ip"], col_map["src_port"], col_map["dst_ip"], col_map["dst_port"],
                           col_map["timestamp"]]

            # 2. 局部列读取，自动处理多编码兼容
            try:
                df = pd.read_csv(csv_path, usecols=needed_cols, engine='c', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, usecols=needed_cols, engine='c', encoding='gb18030')

            df.columns = df.columns.str.strip()
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            # 用 Pandas 高度鲁棒的 C 语言解析引擎整列进行批量 Datetime 转换，自适应任何不补零的月份和天数
            ts_cleaned = df[col_map["timestamp"]].astype(str).str.strip().str.replace("下午", "PM").str.replace("上午",
                                                                                                                "AM")

            if source == "DoH2020":
                # 2020 强制使用年优先进行极速转换
                dt_series = pd.to_datetime(ts_cleaned, yearfirst=True, errors='coerce')
                offset = OFFSET_2020
            else:
                # 2017 使用日优先进行转换
                dt_series = pd.to_datetime(ts_cleaned, dayfirst=True, errors='coerce')
                offset = OFFSET_2017

            # 【核心修正 2】：填充空时间戳，彻底保护 int64 转换时在 Windows 上不闪退
            dt_series = dt_series.fillna(pd.Timestamp('1970-01-01'))

            # 3. 转换并补偿时差
            epochs = dt_series.astype('int64').values // 10 ** 9 - offset
            epochs[epochs < 0] = 0

            # 4. 插入清洗好的统一 Epoch 秒数，删除原始长文本时间列
            df["ts_epoch"] = epochs
            df = df.drop(columns=[col_map["timestamp"]])

            # 5. 标准化更正列名为统一格式，方便 Sampler 高速免索引识别
            df = df.rename(columns={
                col_map["src_ip"]: "src_ip",
                col_map["src_port"]: "src_port",
                col_map["dst_ip"]: "dst_ip",
                col_map["dst_port"]: "dst_port"
            })

            # 6. 保存为干净的新文件
            df.to_csv(output_path, index=False, encoding='utf-8')

            del df
            gc.collect()
        except Exception as e:
            tqdm.write(f"  [解析失败] 无法处理文件 {csv_path}: {e}")
            continue

    print(f"\n[全量日志清洗完毕！] 干净的数据集已保存在绝对路径: {PROCESSED_CSV_DIR}")


if __name__ == "__main__":
    preprocess_all_csvs()