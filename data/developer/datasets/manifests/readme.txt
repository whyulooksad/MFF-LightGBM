存放数据集清单和可复现划分文件：
- source_manifest.json：15,280 个官方 PCAP 的路径、大小、SHA-256、数据集、标签和来源字段。
- split_manifest.json：15,279 个可用 PCAP 的 group_id 与 train/validation/test 固定划分。
- split_report.json：划分数量、类别覆盖和泄漏校验；当前 status=passed、leaking_groups=0。

1 个官方零字节 iodine 抓包保留在 source_manifest 中，标记 usable=false，不进入 split 或 prepared。训练脚本必须读取清单，不能依赖文件名临时猜标签，也不能随机打散 flow 重新划分。
