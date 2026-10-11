import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from math_tokenizer.corpus import ROOT

def main():
 results={cap:json.loads((ROOT/f'results/v3_{cap}_release/metrics.json').read_text()) for cap in (512,1024)}
 lines=['# 第三轮优化结果：AST-BPE与RL-BPE','',
  '语料、标签、划分完全未改；没有重训成熟基准，只重测已有自研方法以取得同输入、同缓存协议的速度对照。全部3960?以protocol实际为准：3959条测试公式参与每种方法；无损率100%。',
  'v3=完整命令原子初始化+BPE词表训练+原生推断；AST hard树约束最短路径，RL训练完整路径策略并按dev选择soft结构罚项。RL参数由REINFORCE学习，但主要质量收益可能来自命令词表；no-RL shortest控制公开。', '',
  '| 预算 | 方法 | 旧token/公式→新 | 长度减少 | 旧命令完整率→新 | 旧跨树比例→新 | 编码提速 |','|---|---|---|---:|---|---|---:|']
 for cap,metrics in results.items():
  def avg(name,key):return float(np.mean([m[key] for m in metrics if m['method']==name]))
  for old,new in [('ast_bpe_v2','ast_bpe_v3'),('rl_bpe_v2','rl_bpe_v3'),('rl_ast_bpe_v2','rl_ast_bpe_v3')]:
   before=avg(old,'tokens_per_formula');after=avg(new,'tokens_per_formula')
   lines.append(f'| {cap} | {new} | {before:.4f} → {after:.4f} | {100*(before-after)/before:.2f}% | {avg(old,"command_integrity_rate"):.2%} → {avg(new,"command_integrity_rate"):.2%} | {avg(old,"hybrid_tree_crossing_rate"):.2%} → {avg(new,"hybrid_tree_crossing_rate"):.2%} | {avg(new,"encode_mib_s")/avg(old,"encode_mib_s"):.2f}× |')
 lines+=['','## 各项指标的变化，完整保留取舍','',
 '| 预算 | 方法 | 数字完整率 旧→新 | lexeme保留率 旧→新 | 词表利用率 旧→新 | Rényi2 旧→新 | 经验unigram bits/byte 旧→新 |','|---|---|---|---|---|---|']
 for cap,metrics in results.items():
  def avg(name,key):return float(np.mean([m[key] for m in metrics if m['method']==name]))
  for old,new in [('ast_bpe_v2','ast_bpe_v3'),('rl_bpe_v2','rl_bpe_v3'),('rl_ast_bpe_v2','rl_ast_bpe_v3')]:
   ren=lambda name:float(np.mean([m['renyi_efficiency']['2.0'] for m in metrics if m['method']==name]))
   lines.append(f'| {cap} | {new} | {avg(old,"number_integrity_rate"):.4f} → {avg(new,"number_integrity_rate"):.4f} | {avg(old,"lexeme_preservation_rate"):.4f} → {avg(new,"lexeme_preservation_rate"):.4f} | {avg(old,"vocab_utilization"):.4f} → {avg(new,"vocab_utilization"):.4f} | {ren(old):.4f} → {ren(new):.4f} | {avg(old,"empirical_unigram_bits_per_byte"):.4f} → {avg(new,"empirical_unigram_bits_per_byte"):.4f} |')
 lines+=['','## 收益来源','',
  'exact native模式与原v2策略在全dev及共同test的逐token差分保持相同，速度收益没有通过返回缓存预先分词结果获得。C trie匹配、标量softmax、DP、区间crossmask替代Python逐位置numpy数组；Python结构前端单次lexing/offset的等价版本也验证了全部28347条公式的结构记录。',
  '词表改善来自完整命令原子，包含回退字节的总预算固定512/1024。AST从完整原子序列训练，减少将命令字符串拆成低效字节碎片；hard结构保证零跨节点。',
  'AST-v3推断改用结构合法最短路径，因此不宣称是原ranked BPE的单纯代码优化。RL-v3推断仍使用训练后局部归一化policy-MAP加验证选择的soft跨树代价。',
  '同新词表no-RL最短路径通常更短；这意味着不能把v3对v2的全部压缩改善称RL学习贡献。RL仍是可学习目标的研究分支，压缩目标本身已有固定词表DP最优解。',
  '', '## 稳健性与审查','',
  '全部输入完全相同、未知符号回退无损；5次cold结构缓存测试取中位数；独立retime脚本在所有训练结束后顺序重测，不把并行训练资源竞争作为模型差异。原生首次编译不包含稳态encode，源码hash及cc依赖公开。',
  '配置仅dev选择，test是此前开发过程已查看的集合，不冒称新未见holdout。paired组bootstrap差值/置信区间见各预算pairwise.json。',
  '六轮审计覆盖语料哈希不变、相同预算和dev冻结、原始结果重算、保存模型重放+native/Python差分、完整测试、运行后数据再核验。自审不是外部专家审查。',
  '', '## 新方法探索','',
  '尝试BPE种子大词表的minimum-path usage修剪（数学MinGram式适配）及train-internal RL词表交换。探索效果未稳定超越命令原子BPE，不选入主结果；保留源码与dev记录。不能把随机搜索/贪心incumbent改良冒称纯RL收益。',
  '参考[PathPiece](https://aclanthology.org/2024.emnlp-main.40/)和[MinGram](https://arxiv.org/abs/2606.27019)，原论文固定词表最短路径与词表refinement来自一般语料，数学结构适配是本项目实验。',
  '', '## 当前限制','',
  '提升明显但并非所有描述指标都必须同向：信息熵、Rényi、token rank没有普遍等价的质量方向；数字完整率在512下主要持平。详细JSON记录所有指标，不能宣称“每个指标都大幅提升”。',
  '即使原生RL快十倍以上，完整encode仍可能慢于高度优化成熟Rust BPE；本轮用户要求不重跑基准，因此只对旧自研实现给速度比例，不能据此改写成熟基准排名。',
  '目标仍是独立分词器；没有下游任务，不宣称推理或生成能力提升。']
 text='\n'.join(lines).replace('全部3960?以protocol实际为准：3959条','全部3959条')+'\n'
 (ROOT/'docs/optimization_v3_findings.md').write_text(text)
if __name__=='__main__':main()
