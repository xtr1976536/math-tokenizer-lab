# v5顺序复现

旧release保留，新增v5语料和holdout。Python依赖使用requirements-v2.lock.txt；RL learned table通过未修改官方tokenizers Rust编码，不需要C编译器或AST parser参与推断。

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-v2.lock.txt
.venv/bin/python -m pytest tests -q
bash scripts/run_v5_all.sh results/new_v5
```

重建新增source快照（保持原version/许可，不重新选择论文）：

```bash
.venv/bin/python scripts/fetch_v5_sources.py
.venv/bin/python -m math_tokenizer.benchmark_v5
.venv/bin/python -m math_tokenizer.generalization_v5
.venv/bin/python scripts/quality_v5.py
```

保存tokenizer使用：

```python
from pathlib import Path
from math_tokenizer.rl_vocab_v5 import RLVocab
m = RLVocab.load(Path('results/v5_1024/models/broad-rl-selected-17.json'))
s = r'\frac{\partial u}{\partial t}=\alpha u'
assert m.decode(m.encode(s)) == s
```

每个model.json含policy、fit/reward IDs、merge action/reward日志，model.tokenizer.json是官方可直接加载的tokenizer资产。两者相同字节词表/merge规则；teacher labels和人工命令清单不参与。全部beta features和reward weights见protocol，不能声称完全无人工设计。

审计与全holdout重放：

```bash
.venv/bin/python scripts/audit_v5.py results/v5_1024
.venv/bin/python scripts/replay_v5.py results/v5_1024
```

Test句子只在dev selection已冻结后加载；极端领域需要读specialized分支才构成未见domain评估，broad分支只是新论文泛化。synthetic stress例子是独立鲁棒性套件，不加进自然公式benchmark或reward。新的v5test在首轮开发后也将成为已见benchmark，将来不能继续声称新的publication holdout。

GitHub master corpus未压缩大小约120MiB，超过单文件上传限制，发布corpus.jsonl.gz无损归档（约15MiB）；相关脚本首次访问自动解压并验证summary中的SHA。训练/验证/测试JSONL仍可直接读取。压缩仅是存储方式，不改任何模型输入。

2026-10-11全新GitHub checkout验证：主corpus.gz自动解压后SHA一致，141项测试通过，六个正式RL checkpoint在12个scenario/suite组合的全部新holdout/stress记录重放一致。数据隔离、reward范围和模型哈希审计通过；证据results/v5_clean_clone_verification.json。复用了独立安装的旧依赖环境，本次未重新训练所有配置或再次下载119份源稿，不冒称跨平台完整重训练复现。
