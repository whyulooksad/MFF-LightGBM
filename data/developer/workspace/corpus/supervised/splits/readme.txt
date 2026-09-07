本目录保留给需要按文件导出的监督 train/validation/test JSONL。
当前主流水线把固定 split 写在 supervised_flows.jsonl 每条记录中，由 Dataset 按字段选择集合；不会在这里重新随机划分。
