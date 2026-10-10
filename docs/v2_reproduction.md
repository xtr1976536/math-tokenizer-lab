# 第二版顺序复现

项目根目录、Python3.11+、uv。已有完整语料、来源registry和保存模型可直接拉取；不需要再次抓arXiv才使用数据。环境锁文件含官方TokEval固定commit，安装的是第三方原版包。

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-v2.lock.txt
.venv/bin/python -m pytest tests -q

# 如需重建全部数据，按已有registry和固定revision，不换最新来源：
.venv/bin/python scripts/fetch_sources.py
.venv/bin/python scripts/fetch_v2_sources.py
.venv/bin/python -m math_tokenizer.corpus_v2
.venv/bin/python scripts/fetch_baselines.py

# 新结果目录，512/1024及审计顺序执行：
bash scripts/run_v2_all.sh results/new_reproduction
```

完成的run目录拒绝覆盖。protocol.json在训练前写入数据ID/哈希、全部候选超参数、验证选择规则；selection.json在test前冻结。实验中断或语料质量修正的探索性记录单列interruption/invalidated，不能混作正式结果。

保存模型的独立使用：

```python
from pathlib import Path
from math_tokenizer.merge_v2 import FastBPE
from math_tokenizer.rl_v2 import LatticeTokenizer

m = FastBPE.load(Path('results/v2_release512/models/base-ast.json'))
s = r'\sum_{i=1}^{n}\frac{x_i}{1+x_i^2}'
assert m.decode(m.encode(s)) == s

rl = LatticeTokenizer.load(Path('results/v2_release512/models/rl_byte_selected-17.json'))
assert rl.decode(rl.encode(s)) == s
```

两档复杂度在 `data/releases/v2/curated/by_domain/<domain>/<origin>/<simple|complex>.jsonl`；纯字符串在同release的 `formula_only/<domain>/<origin>/<class>.jsonl`。领域标签的主题推定依据和复杂度分量同记录保留。

算法训练benchmark与清缓存速度replay顺序执行；不并行训练同时测速度。独立审计 `scripts/audit_v2.py <result_dir>`、官方保存模型重放 `scripts/replay_official_models_v2.py <result_dir>`。审计依赖已下载来源只用于验证源码SHA，测试代码、模型、公式本体不要求下载源全文。数据重建复核 `scripts/verify_v2_rebuild.py`。

所有结果属于本地机器与当前锁定环境，不声称其他操作系统完全相同。时间的自然波动不会影响逐公式token数或固定随机种子训练输出。
