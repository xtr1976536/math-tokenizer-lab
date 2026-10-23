# 下游八案例数据与预处理结果

下游实验的八个原始数据集位于 [`third_party/eqgpt/data/`](../third_party/eqgpt/data/)，来自固定的 [EqGPT 原作者仓库](https://github.com/woshixuhao/EqGPT)。数据格式包括 MAT、CSV 和 NPY，各自原说明文件也在对应案例目录。完整论文原版 EqGPT 程序见 [`third_party/eqgpt/code/`](../third_party/eqgpt/code/)；本项目缩小版运行代码见 [`scripts/eqgpt/`](../scripts/eqgpt/) 与 [`src/math_tokenizer/eqgpt/`](../src/math_tokenizer/eqgpt/)。

| 案例 | 原始数据 | 已拟合 surrogate 缓存 |
|---|---|---|
| Wave | [`Wave_equation`](../third_party/eqgpt/data/Wave_equation/) | [`surrogate_cache.pt`](../experiments/downstream/Wave_equation/surrogate/surrogate_cache.pt) |
| Convection-diffusion | [`Convection_diffusion_equation`](../third_party/eqgpt/data/Convection_diffusion_equation/) | [`surrogate_cache.pt`](../experiments/downstream/Convection_diffusion_equation/surrogate/surrogate_cache.pt) |
| Klein-Gordon | [`KG_equation`](../third_party/eqgpt/data/KG_equation/) | [`surrogate_cache.pt`](../experiments/downstream/KG_equation/surrogate/surrogate_cache.pt) |
| Burgers | [`Burgers_equation`](../third_party/eqgpt/data/Burgers_equation/) | [`surrogate_cache.pt`](../experiments/downstream/Burgers_equation/surrogate/surrogate_cache.pt) |
| Eq. 6.2.12 | [`Eq_6_2_12`](../third_party/eqgpt/data/Eq_6_2_12/) | [`surrogate_cache.pt`](../experiments/downstream/Eq_6_2_12/surrogate/surrogate_cache.pt) |
| PDE divide | [`PDE_divide`](../third_party/eqgpt/data/PDE_divide/) | [`surrogate_cache.pt`](../experiments/downstream/PDE_divide/surrogate/surrogate_cache.pt) |
| KdV | [`KdV_equation`](../third_party/eqgpt/data/KdV_equation/) | [`surrogate_cache.pt`](../experiments/downstream/KdV_equation/surrogate/surrogate_cache.pt) |
| Chafee-Infante | [`Chaffee_Infante_equation`](../third_party/eqgpt/data/Chaffee_Infante_equation/) | [`surrogate_cache.pt`](../experiments/downstream/Chaffee_Infante_equation/surrogate/surrogate_cache.pt) |

每份 `surrogate_cache.pt` 保存了**最终 MLP 参数、供 reward 求导的 10,000 个采样坐标和预测值**。这是已经完成的拟合结果，下载仓库即可复用；不需要重新运行 50,000 次 MLP 训练。缓存的网络配置、数据哈希和拟合质量在同案例的 `surrogate/shared_manifest.json` 与 `surrogate_quality.json` 中。原训练过程中的每 500 次阶段性 `.pkl` 不用于后续搜索，因此没有重复存入仓库。

另外，八案例的 Transformer 权重以私有 GitHub Release 资产提供，文件与张量哈希清单见 [`weights_manifest.json`](weights_manifest.json)。恢复命令和直接加载示例见 [`scripts/eqgpt/README_权重.md`](../scripts/eqgpt/README_权重.md)。这批八案例均为已知控制方程的结构回收测试；新有效方程数据集尚未纳入。
