# 第三轮复现

语料仍使用第二版原始文件和划分，不重建、不再采集；v2成熟基准模型无需下载/重训。Python环境继续用requirements-v2.lock.txt；新增native C源码自动调用`cc -O3 -shared -fPIC`编译，Mac/Linux环境需C编译器。首次编译及词表加载与稳态encode分别处理。

```bash
# 项目根目录
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-v2.lock.txt
.venv/bin/python -m pytest tests -q

# 顺序执行两个预算，输出目录不得覆盖
bash scripts/run_v3_all.sh results/new_v3
```

直接加载可复核的模型：

```python
from pathlib import Path
from math_tokenizer.native_v3 import NativeTokenizer

ast = NativeTokenizer.load(Path('results/v3_512_release/models/ast-atom128.json'))
rl = NativeTokenizer.load(Path('results/v3_512_release/models/rl_bpe_v3-17.json'))
text = r'\frac{\partial u}{\partial t}=\sum_{i=1}^{n}u_i'
assert ast.decode(ast.encode(text)) == text
assert rl.decode(rl.encode(text)) == text
```

审计、编译及差分：

```bash
.venv/bin/python scripts/audit_v3.py results/v3_512_release
.venv/bin/python scripts/audit_v3.py results/v3_1024_release
.venv/bin/python scripts/native_speed_probe.py
```

模型训练选择保存protocol.json、selection.json、dev_ablations.json，实际run的候选值优先于今后脚本默认值。RL质量和编码速度变化要与atomic_byte_shortest_control及exact-native对照读取：后者不改变旧策略的输出，前者不使用RL。

独立重放新模型的测试结果可以直接比较每公式token IDs/数量、exact decode、结构span，速度有自然波动。C binary存在.native_cache，未上传Git；源hash记录在每个模型。语法前端加速版本未改变v2语料标签或复杂度字段。
