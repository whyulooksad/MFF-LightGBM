# Legacy archive

此目录只用于追溯旧实现和旧实验，不属于当前运行路径。

- `code/`：旧的近似 TLS/X.509 特征提取器和逐流截断器。
- `models/production/v1/`：与旧字段契约绑定的历史模型。
- `data/workspace_pre_strict_parser/`：不符合当前特征与划分契约的 54 个旧工作区文件（约 2.22 GB）；大文件由 `.gitignore` 排除，只跟踪说明。
- `docs/README_legacy.md`：旧版长说明与历史指标。
- 其余目录：重构前留下的旧数据准备代码和实验产物。

`developer/` 和 `user_app/` 均不得导入本目录中的 Python 文件。
