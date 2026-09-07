# 模型目录

- `base`：官网下载的原始基础模型，只读。
- `experiments/<run_id>`：开发者训练出的候选模型。
- `production/<release_id>`：通过校验后发布的冻结模型，例如 `2026-09-strict-parser`。
- `production/active.json`：用户系统当前加载的发布标识；`unpublished` 表示尚无兼容模型。

禁止训练代码直接覆盖 `production`。

旧模型已经移动到 `legacy/models/production/v1`。它由近似 TLS/X.509 字段训练，不能与当前严格解析器混用，也不会被用户系统加载。
