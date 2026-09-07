# 开发者制模通道

这里负责数据准备、模型训练、评估和发布。用户上传不会进入这里。

当前两个官方数据集已经完成清单和固定分组划分：15,279 个可用 PCAP 位于 `data/developer/datasets/prepared/`（train 10,084、validation 2,066、test 3,129），均为指向 source 的同盘硬链接。现在可直接运行：

```powershell
uv run python -m developer.pipeline --stage all
```

该命令只执行特征提取到 LightGBM 的默认训练主链；它不重建 manifest/prepared，也不执行 `evaluate`、模型契约构建、校验或发布。

```text
官网下载并原样解压
→ build_manifest（来源、哈希、标签）
→ split_dataset（按数据集 + 类别分层，完整 sample/capture/APK/PCAP 分组）
→ batch_extract（复用用户运行时的唯一提取器）
→ serialize_flow
→ DeBERTa-v3 RTD
→ LoRA 八分类
→ CLS 768 维
→ SupCon-AE 64 维
→ 加 80 维数值特征
→ LightGBM
→ 独立 test 与消融
→ validate_bundle / publish
```

常用命令及目录约定见 `docs/模型训练说明.md`。训练产物只能写入 `models/experiments`；只有通过特征版本、字段顺序、标签顺序和完整性检查的候选包才能发布到 `models/production`。

所有训练阶段必须继承 `group_id` 与 `split`。缺失来源分组、类别没有覆盖三个集合、同组跨集合或尝试用 flow 随机拆分时，代码会直接报错。外层 test 在完整模型链最终评估前保持封存。
