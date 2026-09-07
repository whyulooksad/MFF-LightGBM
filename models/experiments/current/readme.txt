开发者离线训练的当前候选模型写入本目录。
当前还没有兼容严格解析器的完整候选包。运行 developer.pipeline --stage all 会生成训练产物，但不会自动发布。
这里的文件不能被 Web 推理加载；评估、契约构建和校验通过后，才可显式发布到 production/<release_id>。
