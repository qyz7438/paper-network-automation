# 论文网络自动化：研究原型

这是一个面向 Obsidian 论文库的实验性工具，使用本地 PDF 文本、GLiNER2.5 Small 和 multilingual-e5-small 发现方法、问题、任务和数据集候选，并生成供审核的检索与卡片资料。

当前版本**尚未通过 100 篇自动成卡验收**。局部关系抽取诊断曾返回空结果，查询翻译测试亦未通过；程序中的候选提及和相似度不能直接视为论文采用关系或图谱边。正式图谱写入需要另行审阅和验证。

## 安装

使用 Python 3.12，先按本机平台安装 PyTorch CPU 版本，然后安装可选模型依赖：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/paper_pipeline/requirements-local-models.txt
python -m scripts.paper_pipeline.cli --help
```

模型下载是单独的联网准备步骤；模型推理使用本地权重。默认缓存位于用户目录下的 `.cache/llm-wiki`，可用 `LLM_WIKI_MODELS_DIR` 和 `LLM_WIKI_TRANSLATION_MODEL_DIR` 指定其他目录。模型与 PDF 均未包含在此仓库。

实际处理时，应按工作流说明显式传入 `--vault <Obsidian vault 路径>`；默认仓库根目录不是你的论文库。

## 测试与输入

```bash
python -m unittest tests.test_paper_pipeline tests.test_paper_pipeline_local tests.test_paper_workflow tests.test_card_enrichment tests.test_relation_trial tests.test_cache_defaults
```

这些测试使用合成数据或模拟对象，不构成真实论文质量验收。现有批处理入口依赖特定的 Obsidian 库结构、冻结清单和本地解析缓存；仓库不包含这些私人输入，也不能开箱运行现有的100篇试验。详见 `scripts/paper_pipeline/README_WORKFLOW.md`。

## 许可证

本仓库尚未指定许可证。
