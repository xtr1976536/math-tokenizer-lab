# EqGPT 数学分词主实验：第一轮 AutoDL 运行说明

日期：2026-10-16。本包只运行第一轮统一参数的主实验。第二轮调参、容量与语料消融、自动词表及 SAT 改造均不执行。

## 1. 运行内容

| 系统 | 训练表示 | 总 ID | 生成 |
|---|---|---:|---|
| 原 EqGPT | 作者 GRE | 57 | 作者 SAT、80% 模型采样＋20% 合法动作随机探索 |
| 普通 BPE | LaTeX | 1,024 | 全词表自回归，无 SAT |
| 标准 AST-BPE | 相同 LaTeX | 1,024 | 全词表自回归，无 SAT |
| 软 AST-BPE | 相同 LaTeX | 1,024 | 全词表自回归，无 SAT |

三个 LaTeX 系统是同表示、同容量的分词器比较；与原 EqGPT 的比较是包括 GRE/LaTeX 与 SAT 差异的系统比较。

- 作者代码来自 `woshixuhao/EqGPT`，固定提交 `dc35fc0ac860ff224293f01d2a62204042095127`；原文件逐字节保留。
- 六层 Transformer 使用作者原始类：8 头、隐藏维 768、前馈维 2048、每头 K/V 64。词表相关嵌入和输出层适配；四组共用 512 个位置、相同主干初值。GRE 科学动作长度条件仍为 50。
- 所有 Transformer 均从零初始化，不加载 GPT-2 或其他预训练语言模型。
- 新词表已在固定广领域 benchmark 训练分区的 **32,270 条、13 个领域**公式上训练并冻结。原候选训练分区 36,615 条，其中 4,345 条被筛除或隔离；验证/测试分区 19,385 条未用于词表训练。逐条清单、来源、隔离原因与哈希附于 `experiments/eqgpt_round1/frozen_math/`。
- Transformer 仅在作者 221 行 PDE 手册经当次目标排除后的公式上训练，采用作者 32 倍增强。通常 218–220 条原式、约 7,000 条增强式。数学 benchmark 不用于 Transformer 预训练。
- 第一轮四组均 Adam、LR=1e-4、100 epoch、batch=128 条公式、梯度裁剪 1、常数学习率、FP32、关闭 TF32。
- 11 案例 × 种子 525/526/527 × 4 系统 = **132 组最终模型**；每组 5 周期 × 400 次生成尝试。前三个 LaTeX 系统 EOS 停止、最多新生成 256 token / 2,048 原始字节，完整分布采样，temperature=1。原 EqGPT 按作者动作限制。
- 无 SAT 不等于修改后续数学评估器。完成生成后，安全解析器仅将作者支持的 LaTeX 项映射回原 GRE 数值接口；不支持的合法数学表达式也单列记录，不强行修复。
- 奖励、回归、候选 Top-10、继续训练 5 epoch / LR=1e-5、PINN 前 3 候选 / 500 步 / data+0.01×PDE 均调用作者函数。候选不足 10 个按实际数量更新；0 个跳过更新进入下一周期；不补采样。
- 替代网络每个案例/种子只训练一次并冻结共享快照；作者默认训练 50,000 步，四组复用。前 10 案例 10,000 观测点，圆盘 Poisson 30,000 点，噪声 0。

实际工作量见 `docs/eqgpt/round1_workload.json`：约 726,000 次 Transformer 预训练更新、1,650,000 次共享替代网络更新、264,000 次生成尝试。

## 2. AutoDL 配置

建议单张 RTX 4090 / 4090D 24 GB，至少 8 vCPU、32 GB RAM，数据盘可用空间 **100 GB 以上**。优先 Python 3.11、CUDA 12.1 开发镜像，带 gcc / Python 开发头文件。保留作者科学计算环境：PyTorch 2.3.1、NumPy 1.26.4、SciPy 1.13.1 等，完整列表见 `requirements-eqgpt.lock.txt`。

本包不采用 BF16、AMP、TF32、FlashAttention、KV cache 或架构替换。此锁定版本不支持直接选择 RTX 5090 等 Blackwell 卡。4090 的 FP32 性能与价格通常适合这一规模；GPU 实际显存峰值由短测记录。

## 3. 上传、安装与检查

将 `eqgpt_round1_autodl_20261016.zip` 上传到 `/root/autodl-tmp`，打开终端：

```bash
cd /root/autodl-tmp
unzip eqgpt_round1_autodl_20261016.zip
cd eqgpt_round1_autodl_20261016
bash scripts/eqgpt/install_round1.sh
```

安装程序创建独立 `.venv-eqgpt`，安装锁定版本，检查归档逐文件 SHA-256、作者原文件、31 个数据文件、冻结词表、CUDA 前反向和 C 扩展，并运行 33 项测试。原始源码和数据已包含，无需再次下载 EqGPT 或私有 GitHub 仓库。

如镜像中的默认 Python 不是 3.11，请指定现有 3.11 可执行文件后再安装：

```bash
EQGPT_PYTHON=/path/to/python3.11 bash scripts/eqgpt/install_round1.sh
```

可先做云端缩小预算的流程测试：

```bash
PYTHONPATH=src .venv-eqgpt/bin/python scripts/eqgpt/run_latex.py \
  --device cuda --smoke --attempts 400 --output cloud_smoke
```

烟雾测试四组各训练 1 epoch、4 个共同公式、每组 2×400 次生成，缩小替代网络和 PINN 步数。它用于检查接口，不进入正式结果。

## 4. 短测时间与费用

```bash
PYTHONPATH=src .venv-eqgpt/bin/python scripts/eqgpt/calibrate_round1.py \
  --hourly-price 2.0 --output gpu_calibration.json
```

`2.0` 是费用测算示例，请替换为租用页面的实际元/小时。短测在真实原始公式批次上测四个模型的前反向、不同前缀长度的生成前向、作者替代网络的训练步及峰值显存，不保存为正式模型，不选择超参数。约数分钟，结果写入 `gpu_calibration.json`。端到端预估另含生成长度、CPU 最小二乘、验证及候选高阶导数/PINN 的成本，首个完整案例完成后再用其真实阶段耗时校准。

尚无本次云端实测前，对 **全部 132 组**先按单张 4090 **20–50 小时**准备；无 SAT 输出常达长度上限时，生成和不支持表达式的成本会增加，可按 **60 小时**留余量。以每小时 2 元计算，主体计算费约 40–100 元，60 小时为 120 元；以每小时 3 元计算约 60–150 元，60 小时为 180 元。此处时薪是预算情景，不是当前 AutoDL 报价。

单个案例×1种子×4系统可先按 0.6–1.5 小时准备；不同方程的高阶导数成本不同，不代表每个案例等时。

平台按实例开机时间计费，安装/空闲也计时；扩容盘另外计费。价格与计费规则以 [AutoDL 官方计费说明](https://www.autodl.com/docs/price/) 和租用页面为准。建议首轮准备 150–200 元预算余量，并在得到短测与首个正式案例耗时后收窄预算。

## 5. 正式运行：只运行第一轮

可先查看完整计划，不训练、不创建输出结果：

```bash
PYTHONPATH=src .venv-eqgpt/bin/python scripts/eqgpt/run_round1.py \
  --output /root/autodl-tmp/eqgpt_round1_results --plan-only
```

运行全部 33 个配对任务：

```bash
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-eqgpt/bin/python scripts/eqgpt/run_round1.py \
  --output /root/autodl-tmp/eqgpt_round1_results
```

建议在 `tmux` 内运行，避免浏览器断线中断。程序顺序运行，不自动申请/购买 GPU，不自动关机或删除结果。只验证单个正式配对任务时可指定 `--cases KdV_equation --seeds 525` 并用独立输出目录；完整运行使用上面的完整命令，不将不同计划混进同一目录。

每个任务的输出包括：

```text
eqgpt_round1_results/
  plan.json
  KdV_equation/
    seed525.log
    seed525/
      RUN.json
      manifest.json
      timing.json
      shared/                        # 固定观测、增强清单、替代网络
      bpe/                           # pretrained.pt / searched.pt / training.json
      ast_bpe/                       # search.json 含全部原始尝试
      soft_ast_bpe/                  # pinn.json 含系数、采样哈希、状态
      eqgpt_reference/
  ...
  latex_results.json
  latex_results.csv
```

## 6. 中断恢复与汇总

重新运行同一条正式命令即可：

- 已完成任务先核对源代码/词表哈希、模型检查点及 2,000 次尝试，再跳过。
- 预训练每 10 epoch 保存模型、Adam 与随机数状态，从最近保存的完整 epoch 恢复。
- 已完成的替代网络、预训练、发现、PINN 阶段复用各自结果；未完成的发现或 PINN 阶段从该阶段起点重跑。
- 如果替代网络训练尚未完成，重新执行该共享阶段。不是任意迭代粒度的恢复。
- 配置/代码变化会拒绝混用旧目录。

单独汇总正式结果：

```bash
PYTHONPATH=src .venv-eqgpt/bin/python scripts/eqgpt/collect_latex.py \
  /root/autodl-tmp/eqgpt_round1_results
```

汇总保留无可用候选和数值失败，以全部尝试/任务为分母；输出结构恢复、Top-10 恢复、可数值评估比例、失败计数、系数误差及成本。烟雾实验默认排除。

## 7. 本地审查证据

`docs/eqgpt/round1_evidence/audit.json` 记录四轮审查：

1. 独立数学训练分区、目标隔离、冻结词表、51 基本项/221 公式数学往返与作者源哈希。
2. 主干初值、实际反向、优化器绑定、完整词表无 SAT、真实作者数值/候选/PINN 接口的自动测试。
3. 全部 11 案例、44 个从零模型、35,200 次实际烟雾生成，检查失败计数、继续更新、数据共享和重载 logits。
4. 独立重复 KdV，除耗时外训练损失/参数/候选/奖励/PINN 记录完全一致；中断后的模型/优化器/随机数恢复与连续训练一致。

`smoke_records.jsonl.gz` 保存对应完整记录。这些是缩小预算的运行验证，不是正式发现效果。当前本地为 macOS CPU；CUDA 的运行检查由 AutoDL 安装后的 `preflight.py --require-cuda` 和云端 smoke 完成。
