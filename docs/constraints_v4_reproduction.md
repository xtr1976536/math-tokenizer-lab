# 第四轮复现

语料和基准不改，Python依赖沿用requirements-v2.lock.txt。源码native/decoder_v4.c由本地C编译器自动编译，二进制未加入仓库。

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-v2.lock.txt
.venv/bin/python -m pytest tests -q
bash scripts/run_v4_all.sh results/new_masked_run
```

加载主RL策略：

```python
from pathlib import Path
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
m = ConstrainedTokenizer.load(Path('results/v4_1024/models/rl_v4-17.json'))
s = r'\frac{\partial u}{\partial t}=\sum_{i=1}^{n}u_i'
assert m.decode(m.encode(s)) == s
```

`prepare(s)`返回legal-boundary mask、tree spans和exceptions。covered command/codepoint不可在内部切边界；未知token缺完整词表条目时只走显式byte escape。这里atom unit表示单个命令/Unicode符号，不是把整个任意新公式伪装为一个词表ID。

严格零cross只对共同parser生成的source spans保证。semantic AST与LaTeX CST分开记录，不能以CST高覆盖宣称全公式语义解析正确。formal scalar metrics使用原metrics_v2统一计算，不按mask自定义更宽松metric。

all_configs、seed和训练期望reward曲线保存。expected_policy_gradient包含真实policy参数优化，没有shortest-path标签或下游模型，原算法的score-function思想保持。推断greedy选学习到的logit最高合法动作，policy_map另列dev消融。

可直接执行checkpoint重放和审计，不必重训练：

```bash
.venv/bin/python scripts/replay_v4.py results/v4_512
.venv/bin/python scripts/replay_v4.py results/v4_1024
.venv/bin/python scripts/audit_v4.py results/v4_512
.venv/bin/python scripts/audit_v4.py results/v4_1024
```

原v3代码和模型不覆盖，严格结构版和压缩版可以并存。当前test已参与前轮开发，新的publication holdout仍未采集（用户要求冻结语料），所以本轮不称最终未见泛化实验。

训练耗时字段当前train_seconds针对policy training；词表训练成本在ast模型中记录。与前轮比较策略训练时间时不能把不同范围当总训练时间。两种budget各123条dev记录（120 policy/seed/mode记录+3 AST），formal测试加匹配初始化控制共14个方法/seed、55426条逐公式记录。

2026-10-11全新GitHub clone验证通过125项测试，native_v4从clone源码编译而未复制二进制。两套各9个主模型/控制全test重放一致，另外72个候选checkpoint分层抽样与协议审计通过。补充matched-initial控制也在原目录完成全test重放。证据results/v4_clean_clone_verification.json；这次验证保存模型推断，不是独立重训练全部候选。
