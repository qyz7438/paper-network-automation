# 论文网络自动化：研究原型

这是一个面向 Obsidian 论文库的实验性工具，使用本地 PDF 文本、GLiNER2.5 Small 和 multilingual-e5-small 发现方法、问题、任务和数据集候选，并生成供审核的检索与卡片资料。

当前版本**尚未通过 100 篇自动成卡验收**。关系抽取的空结果诊断发现原测试中部分候选分数低于阈值、其余样本未生成候选对；短关系标签在合成控制中有所改善，但不能推断已解决真实论文抽取。查询翻译测试亦未通过。程序输出的实体、关系和相似度都是待核验候选，不能直接作为论文事实或图谱边。

新增实验阶段包括按原句生成“提出、采用、比较、背景、训练和评测”等角色候选、扩展正文证据窗口、作者元数据与首页对照，以及独立的本地小模型二次复核。二次复核仍会漏掉明显错误，不能自动批准写图。固定测试需要使用方自己提供可追溯的本地论文、冻结清单和人工参考数据；本仓库不含这些输入。

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
python -m unittest tests.test_paper_pipeline tests.test_paper_pipeline_local tests.test_paper_workflow tests.test_card_enrichment tests.test_relation_trial tests.test_cache_defaults tests.test_automatic_roles tests.test_automatic_pilot tests.test_audit_automatic_pilot tests.test_cold_pilot tests.test_reclassify_fresh tests.test_reclassify_fulltext tests.test_fulltext_cues tests.test_fulltext_pilot tests.test_metadata_author_candidates tests.test_role_verifier
```

这些测试使用合成数据或模拟对象，不构成真实论文质量验收。现有批处理入口依赖特定的 Obsidian 库结构、冻结清单和本地解析缓存；仓库不包含这些私人输入，也不能开箱运行现有的100篇试验。详见 `scripts/paper_pipeline/README_WORKFLOW.md`。

## 许可证

本仓库尚未指定许可证。
