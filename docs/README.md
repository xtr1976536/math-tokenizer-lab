# Hard AST-BPE 分词可视化

公开网页：https://xtr1976536.github.io/ast-bpe-visualizer/ 。也可直接打开 [index.html](index.html)，不需要服务器。输入 LaTeX 公式，或从 1500 条 PDE benchmark 中搜索并选择公式；可切换 280、315、350 ID 的冻结 Hard AST-BPE 词表，单步、拖动进度或自动播放合并过程。新增“训练规则树 / 编码结构树 / 解码树”三个视图；训练视图展示冻结词表的规则分解，不伪称复现未保存的全语料候选频率。

`data/hard-*.json` 来自八案例 Wave 运行中保存的冻结词表；`data/pde1500.json` 来自上游固定公式集。`data/离线数据.js` 是两者的离线页面打包，运行 `PYTHONPATH=src python scripts/build_visualizer_data.py` 可以重新导出。分词过程使用相同的 UTF-8 字节、学习到的合并次序和结构区间约束；随机 100 条 benchmark 公式在三个词表上的最终 token ID 与 Python 实现全部一致。

用户输入支持上述词表覆盖的 LaTeX 字节，遇到未覆盖字节会显示错误。网页展示原始公式文本及 token 片段，不执行 TeX 宏，也不判断公式是否数学正确。

公开站点不含第三方脚本、统计服务或外部字体请求；用户输入只在浏览器内处理。选取 benchmark 公式时展示论文题名、原文链接和来源记录中的许可状态。多数来源许可尚未确认，网页不将这些短公式视为已获得全文再分发授权。
