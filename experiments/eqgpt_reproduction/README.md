# EqGPT 作者架构：本地校准记录

本目录的所有运行均为 CPU 缩小预算 smoke，不能据此判断模型优劣。
最终校准环境为 PyTorch 2.3.1、NumPy 1.26.4、pandas 2.2.2、SciPy 1.13.1。

- `validated_smoke/`：11 个案例 × 3 个分词器，33 次完整流程执行。
- `validated_repeat/`：KdV 三组重复执行，训练／搜索／PINN JSON 与前次完全一致。
- `noisy_smoke/`：KdV 50% 高斯噪声 smoke。
- `extended_smoke/`：笑脸和 EITech 公开代码执行检查；不纳入当前正式预设。
- `case_data_review.json`：16 个公开数据案例的加载检查，包括两个 Figshare 大文件。
- `preflight_local.json`：科学环境、作者源码／数据哈希和原生 C 分词器检查。
- `docs/eqgpt/audit_summary.json`：四轮审查的逐项证据。

单次 smoke 保留原模型的完整 6 层／768 维结构；缩小的仅是训练样本与迭代预算：
共同前 4 条训练序列、1 个生成模型 epoch、2 次替代网络更新、25 个元数据点、
2 轮搜索、每轮 4 个候选、每轮 1 次继续训练、每个 PINN 候选 2 次更新。
原始词表仍从完整、排除目标后的 32 倍增强训练序列学习，三组相同。

原始作者文件没有修改。所有执行适配和发现的问题见 `docs/eqgpt/README.md`
与 `docs/eqgpt/changes.json`。参数检查包括 embedding/output 大小、共同主干初值、
全部参数可训练、实际反向传播更新、优化器绑定、checkpoint 重载和数值点哈希。

正式 CUDA 结果将写入单独的 `experiments/eqgpt_cloud` 或用户指定目录。
