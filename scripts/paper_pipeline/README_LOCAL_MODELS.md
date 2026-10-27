# 本地模型

依赖见 `requirements-local-models.txt`。抽取器为 GLiNER2.5 Small，向量模型为 multilingual-e5-small；模型准备与本地推理分离。使用 `LLM_WIKI_MODELS_DIR` 和 `LLM_WIKI_TRANSLATION_MODEL_DIR` 可覆盖默认缓存位置。仓库不包含权重，首次获取权重需要网络。

当前模型输出是来源绑定的候选，不直接构成已验证关系。查询翻译试验未通过质量检查，不应视为可自动接受的结果。
