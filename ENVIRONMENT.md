# 环境安装与复现范围

两套方案的核心代码、数据、词表、配置和结果保持原样。新增模型工具只负责加载已保存的权重；它不替代正式实验脚本，也不将当前环境等同于全部历史实验的环境。

## 文件各自代表什么

| 文件 | 身份和用途 |
|---|---|
| `requirements.lock.txt` | 原有上游环境参考，保留原内容；不包含完整下游或benchmark构建依赖。 |
| `requirements-eqgpt.lock.txt` | 原有作者科学计算栈版本参考，保留原内容；不是当前本机运行环境的逐项快照。 |
| `requirements-runtime.lock.txt` | 2026-10-29核验时观察到的直接依赖版本，覆盖两套方案、测试和来源解析。虽然延用lock命名，它不是跨平台、带哈希的完整传递依赖锁。 |
| `reports/repository_two_studies/maintenance/observed_environment.json` | 维护前的本机Python、平台与已安装主要依赖快照；不能证明每次历史运行均使用这些版本。 |
| `reports/repository_two_studies/maintenance/isolated_environment.txt` | 按runtime文件新建独立环境后的完整已安装包列表，用于记录本次验证环境。 |

不要依次安装两个历史版本文件来覆盖依赖；从新的虚拟环境开始。本次验证平台为macOS arm64、Python 3.11.15、PyTorch 2.14.1。CPU最小运行与安装核验在独立环境完成；CUDA、其他系统及其他PyTorch构建未在本次验证，不承诺随机采样或浮点结果逐位相同。

## 推荐安装入口

需要Python 3.11和本机C编译器（macOS可使用Xcode命令行工具）。原生分词组件首次加载时编译。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-runtime.lock.txt
# 在仓库根目录运行；同时允许scripts模块及src包导入
PYTHONPATH=.:src .venv/bin/python -m pytest -q
```

也可以使用uv创建Python 3.11环境并通过`uv pip install --python .venv/bin/python -r requirements-runtime.lock.txt`安装。若仅阅读JSON、CSV和Markdown报告，无需安装或下载权重。

源码包安装元数据`pyproject.toml`提供基础分词依赖和历史stage2额外依赖；它不替代此处完整实验环境。模型工具需要torch和作者实现导入的科学计算依赖。来源重建额外使用pandas、BeautifulSoup、requests；上游基线使用tokenizers、sentencepiece、tiktoken。依赖已包含在runtime文件中。

## 两套方案的使用入口

- [方案一模型：50-token](scripts/eqgpt/README_权重.md)。
- [方案二模型：80-token](scripts/eqgpt/README_方案二模型使用.md)。
- [两套实验代码、原始数据、词表与复现对应](reports/实验方案索引.md)。

完整训练/搜索会有计算开销，并可能写入指定输出；日常检查应优先使用已保存报告、只读核验及模型inspect/generate/score。新工具search要求新的输出目录，拒绝写入两套已归档实验及源数据目录。不要通过修改原审计文件来绕过代码或数据哈希不匹配。
