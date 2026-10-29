# 环境与模型入口维护核验

本次仅新增环境参考、说明和方案二模型使用工具。两套科学实验的原始代码、配置、数据、词表、训练与搜索结果不重写。

`protected_files.json`在改动前记录基线提交与1291个受保护文件SHA-256；三轮检查均重新比对。`checks.json`汇总三轮：独立环境60项测试；两套上游指标、64组下游结果、72份权重核验；文档链接、Git交付和两份Release检查。

`observed_environment.json`是维护前本机环境，`isolated_environment.txt`是按新runtime依赖安装的独立环境；后者通过兼容性检查。环境观察不证明全部历史运行均使用相同传递依赖。

`model_smoke.json`记录CPU上两套Wave各四种模型的加载与生成。`usage_smoke.json`记录新工具的inspect、score及临时目录小预算搜索（3轮×4次），不作为新的正式实验结果。MPS/CUDA和跨平台采样未在本次重新验证。

复查命令：

```bash
PYTHONPATH=.:src python scripts/verify_repository_maintenance.py --round 1
PYTHONPATH=.:src python scripts/verify_repository_maintenance.py --round 2
PYTHONPATH=.:src python scripts/verify_repository_maintenance.py --round 3
```
