存放数据集清单和可复现划分文件，例如 flows.csv。
清单应记录原始文件、数据集、标签、capture_id、family/resolver、group_id 和 train/val/test split。
训练脚本必须读取清单，不能再依赖文件名猜测标签。
