# 用户检测系统

用户只做一件事：上传 `.pcap` 或 `.pcapng`，等待系统返回逐流八分类结果。页面不展示训练 pipeline、脚本名、测试集分数或开发目录。

```powershell
uv run uvicorn user_app.backend.server:app
uv run python -m user_app.inference.pipeline --pcap <文件> --output-dir <任务目录>
```

输入和结果只写入 `data/runtime`。运行时只加载 `models/production/active.json` 指向、且通过特征契约校验的完整模型包。当前没有兼容新解析器的已发布模型时，检测会明确报错；开发者需要先完成重新训练和发布。

```text
原始 PCAP/PCAPNG
→ 链路/IP 解码与 IP 分片重组
→ 双向流与 TCP sequence 重组
→ TLS/X.509 严格解析 + 80 维统计特征
→ 结构化事件序列
→ DeBERTa-v3 + LoRA 的 CLS
→ SupCon-AE
→ LightGBM 八分类
→ predictions.csv + summary.json
```
