# v8复现与生产训练

Python及第三方路径沿用v6设置文档，锁定依赖见`requirements-v8.lock.txt`。作者PathPiece和TokEval的vendor提交不改，KaTeX版本、成熟tokenizer官方下载与源码哈希沿用固定注册。C/C++扩展由系统编译器在`.native_cache`内根据源码哈希构建，不上传机器二进制。

顺序重放已经冻结的本征实验：

```sh
sh scripts/run_v8_replay.sh
```

这条命令执行共同公式评分、七轮计时、八轮结果协议检查、排名与配对区间、下游注册表/诊断重载核验及报告/图。它不会重新封存模型、根据测试选超参数或下载新公式。v8现已评分，复跑不称新未见测试。

保存的`results/v8/verification.json`来自本机验证，不伪装为第二硬件或独立团队复现。记录运行环境限制。重跑测速受机器负载影响，因此排名随实测速度可变化；公式ID、payload、长度与结构计数应完全一致。

生产非RL训练配方：

```sh
.venv/bin/python scripts/train_production_v8.py --scenario broad --capacity 1024 --verify-against-frozen
```

可以分别选择`broad`/`specialized`及`1024`/`2048`。训练得到共同fit词法原子，原版HF BPE使用同允许资料并导出词表，再构建固定tree=.25、boundary=.3的AST路径。未使用RL策略更新、随机搜索或测试奖励。默认输出`results/v8/production_retrain`，禁止覆盖冻结模型。四个场景的实际重训词表及探针ID已核对，与正式版本一致。HF BPE训练不同环境若有频率平局排序差异，核验会明确失败，不能静默替换冻结词表。

下游加载示例：

```python
from math_tokenizer.final_models import load_final
model = load_final('ast_boundary', scenario='broad', capacity=1024)
formula = r'\frac{\partial u}{\partial t}=\Delta u'
assert model.decode(model.encode(formula)) == formula
```

`models/final_intrinsic/registry.json`保留每个精确检查点、SHA、场景/容量、官方接口与角色。RL诊断默认拒绝，明确消融复核才传`include_diagnostics=True`。v6 `load_current`继续作为历史兼容入口，不是最终生产推荐。

若复现学习机制，从`controlled_terminal_v8.py`、`train_terminal_full_v8.py`开始。它们只读原来train/dev范围，并缓存每次训练内候选策略。逐算法17候选×2任务，保留初始化、首批同随机数、终端回报各分量、RLOO梯度、CEM均值、完整fit导出时间。默认恢复已保存policy；若重新训练，请在另一个输出副本保存，不覆盖用于正式结果的检查点。绝不在评分之后运行freeze脚本重新挑模型。

新论文来源由缓存优先、单连接至少4秒间隔的官方arXiv抓取产生，遇403/429停止，2020年前固定版本、CC BY/CC0与原稿核验；原始完整论文缓存不发布。公开formula archive含详细许可与位置证据。语料模板排除和自动数学语法审查不等于专家语义判定。80条双审材料答案为空。

下游尚未执行；候选名单及下一阶段公平性要求见`v8_downstream_handoff.md`。
