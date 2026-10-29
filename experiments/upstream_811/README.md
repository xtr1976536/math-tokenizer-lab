# 方案二：8:1:1 上游实验

本实验与 `experiments/expanded_downstream_80_v1/` 的扩大语料下游共同构成方案二。方案一的三划分上游仍保存在 `experiments/upstream_main/` 和 `experiments/upstream_vocab_size/`，其下游是 `experiments/downstream/`，见[两套方案总索引](../../reports/实验方案索引.md)。

## 数据与选择

完整保留 `data/formulas/pde/pde_equations.jsonl.gz` 的1500条原始LaTeX，不做GRE筛选或八案例排除。依来源、模板和规范化哈希形成489个连接组，固定种子20261028，划分为1200训练、150验证、150测试。精确ID和规则在 `protocol.json`；这些哈希不能证明所有数学等价式均已隔离。

6种基础表协议×22种总容量，共132个组合：78个完成训练和验证，44个训练字符覆盖不足，10个容量小于基础表加控制符。总容量包含PAD/EOS/BOS三个ID。所有词表只在训练集学习；验证集确定全覆盖Pareto配置，另保留预先声明的ascii98/256参照。冻结的3个Hard配置和3个通用分词器才在测试集评估，未在测试集扫描全部容量。

`latex_ascii`与`train_observed`本次恰好都是91字符且学得词表相同，后者是训练观测字符的诊断方案，不能当作独立固定基础表的重复证据。数值结论和限制见[报告](../../reports/preparation_811/实验结果与限制.md)。

## 文件

- `protocol.json`：源数据哈希、分组划分、容量和基础表定义。
- `validation_results.json`、`validation_summary.csv`：全部132个组合的结果及不可运行原因。
- `<alphabet>/<size>.model.json`与`.model.backend.json`：配套词表；`.train.jsonl.gz`与`.validation.jsonl.gz`保存逐公式指标，`.tokens.csv`保存使用次数。
- `selection.json`：测试前冻结选择；`test_results.json`和`*.test.jsonl.gz`保存实际测试结果。
- `audit.json`、`independent_audit.json`：原运行与独立审计；`encoding_timing.json`是纯编码计时，非训练耗时。

## 运行与复核

在仓库根目录按[环境说明](../../ENVIRONMENT.md)安装依赖后运行：

```bash
PYTHONPATH=.:src python scripts/run_upstream_811.py
```

此命令运行/恢复完整实验并重写汇总；希望只核验已保存结果时运行：

```bash
PYTHONPATH=.:src python scripts/verify_two_studies.py --round 2
```

三项指标沿用方案一：源字节数/输出token数、完整LaTeX命令比例、完整数字串比例。不要混用两套测试划分或词表。方案二下游不使用本实验选出的512-ID词表，仍单独使用280/315/350-ID、98基础字符的下游词表。
