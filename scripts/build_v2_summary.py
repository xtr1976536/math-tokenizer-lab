"""Generate cross-budget findings directly from frozen results."""
import json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl


def main():
    corpus=json.loads((ROOT/'data/releases/v2/curated/summary.json').read_text())
    runs={cap:ROOT/f'results/v2_release{cap}' for cap in (512,1024)}
    metrics={cap:json.loads((path/'metrics.json').read_text()) for cap,path in runs.items()}
    lines=['# 第二阶段：语料扩充与独立分词器优化结果','',
        f"第二版保留{corpus['count']:,}条公式，首版1713条，约{corpus['count']/1713:.2f}倍。人工历史来源{corpus['origin_counts']['human']:,}条；98篇pre2020固定版本arXiv论文，逐版本许可为CC BY或CC0。Fungrim使用原版MIT符号转LaTeX渲染；unknown来源单列。",
        f"训练/验证/测试={corpus['split_counts']['train']:,}/{corpus['split_counts']['dev']:,}/{corpus['split_counts']['test']:,}；simple/complex={corpus['complexity_counts']['simple']:,}/{corpus['complexity_counts']['complex']:,}。主题域共13种，显式PDE记录494条。主题桶不等于每个公式数学类型：ode主题来自math.DS，含动力系统辅助表达式，标签解释见arxiv_collection_v2.md。",'',
        '| 方法 | 512 token/公式 | 1024 token/公式 |','|---|---:|---:|']
    names=['byte_bpe','byte_shortest_path','byte_policy_no_rl','rl_bpe_v1','rl_bpe_v2','ast_bpe_v1','ast_bpe_v2','ast_shortest_path','ast_policy_no_rl','rl_ast_bpe_v1','rl_ast_bpe_v2','hf_byte_bpe','sentencepiece_bpe','sentencepiece_unigram','utf8_bytes']
    def mean(cap,name,key):
        return float(np.mean([m[key] for m in metrics[cap] if m['method']==name]))
    for name in names:lines.append(f'| {name} | {mean(512,name,"tokens_per_formula"):.4f} | {mean(1024,name,"tokens_per_formula"):.4f} |')
    lines+=['','成熟固定词表参照另列：','', '| 参照 | 实际词表 | token/公式 | 无损率 |','|---|---:|---:|---:|']
    for m in metrics[512]:
        if m['comparison_group']=='pretrained_reference':lines.append(f"| {m['method']} | {m['vocab_actual']} | {m['tokens_per_formula']:.4f} | {m['roundtrip_rate']:.4%} |")
    lines+=['','## RL效果与控制','']
    for cap in runs:
        b=mean(cap,'byte_bpe','tokens_per_formula');r=mean(cap,'rl_bpe_v2','tokens_per_formula');nr=mean(cap,'byte_policy_no_rl','tokens_per_formula')
        a=mean(cap,'ast_bpe_v2','tokens_per_formula');ra=mean(cap,'rl_ast_bpe_v2','tokens_per_formula');na=mean(cap,'ast_policy_no_rl','tokens_per_formula')
        lines.append(f'{cap}预算：RL-BPE相对原BPE序列减少{100*(b-r)/b:.3f}%，相对相同词表/无RL先验减少{100*(nr-r)/nr:.3f}%；RL-AST相对AST-BPE减少{100*(a-ra)/a:.3f}%，相对无RL先验变化{100*(na-ra)/na:.3f}%。')
        selection=json.loads((runs[cap]/'selection.json').read_text());lines.append(f'配置由dev独立选择：{selection["selected"]}。')
    lines+=['','每预算byte10配置+AST9配置、各2个dev种子，38个完整消融；总计76个dev配置/种子run和46个正式test方法/种子run。RL正式种子17/29/43；所有epoch reward、policy参数、梯度采样方差均保存。',
        'RL-BPE改善必须与固定BPE词表shortest-path参照联合解读。全局最少token路径已利用同词表可选路径，不能把比BPE短的全部收益算作RL贡献。hard结构分支多个配置没有改变dev路径，无RL控制尤其重要。',
        '512中RLOO样本梯度方差明显低于naive（同epoch实际日志，但最终策略并不相同，此为描述性比较，不是固定策略下的无偏方差估计），但更低方差不保证泛化更好；1024配置由dev重新选择，不能把512的最优设置直接称通用最优。',
        '', '## 结构、来源、复杂度和指标','',
        '| 预算 | 方法 | 命令完整率 | lexeme保留率 | 跨树token比例 | Rényi α=2 | 词表利用率 |','|---|---|---:|---:|---:|---:|---:|']
    for cap in runs:
        for name in ['byte_bpe','rl_bpe_v2','ast_bpe_v1','ast_bpe_v2','rl_ast_bpe_v2']:
            rr=[m for m in metrics[cap] if m['method']==name]
            lines.append(f'| {cap} | {name} | {mean(cap,name,"command_integrity_rate"):.4f} | {mean(cap,name,"lexeme_preservation_rate"):.4f} | {mean(cap,name,"hybrid_tree_crossing_rate"):.4f} | {np.mean([m["renyi_efficiency"]["2.0"] for m in rr]):.4f} | {mean(cap,name,"vocab_utilization"):.4f} |')
    lines+=['','全语料结构覆盖：semantic AST '+str(corpus['structure_counts']['semantic_ast'])+'；LaTeX syntax tree '+str(corpus['structure_counts']['latex_syntax_tree'])+'；byte fallback '+str(corpus['structure_counts']['byte_fallback'])+'。语法树不是语义AST。大幅压缩改善主要来自不再让无法语义解析的合法TeX全部逐字节回退。',
        '两档复杂度、来源和领域分层结果在每预算完整report.md。未进入test的主题/来源组合不提供测试数字；单源的小类别不能外推。',
        '新增30余项输出覆盖压缩/尾部长式、词表频率、Rényi/条件熵、UTF8完整性、命令/数字/运算符/树边界、实际运行速度和组bootstrap。原版TokEval信息论指标直接调用，其余公式指标有独立定义。',
        '', '## 审查与可复现性','',
        '92项测试；六轮审查：公式/来源/复杂度；共同测试与dev选择冻结；保存53个checkpoint跨170个分层公式重放；算法/property/梯度与异常测试；98份源码版本许可/哈希；38个dev消融日志。另有所有官方保存基准对共同test重放与独立冷缓存测速。',
        '旧v1算法原文件未改；新增incremental实现与旧扫描策略/RNG训练严格等价的差分测试，1024旧方法控制用加速实现。算力和实现语言不同，训练秒数记录不能冒称同算力。',
        '发布前从全新GitHub checkout和独立Python依赖环境的复现证据以clean_clone_v2_verification.json为准；包括测试、两套完整实验的顺序重跑及逐公式结果对照。',
        '', '## 局限与后续','',
        '2020前历史人工来源遵循用户规则，不是逐作者无自动化辅助认证。数学正确性和定理成立条件未作为分词任务验证；formula-only字符串须通过来源回看上下文。',
        '修复28条v1继承记录绕过v2filter的问题；修正前partial test曾被查看，invalidated.json保留。正式配置仍按dev选择，校正规则没有依赖方法分数，但不能称这是从未接触的最终出版test。后续应新增完全未访问的论文test验证，尤其做方法迭代时。',
        'RL初版是固定BPE词表上的路径策略，未联合优化词表；压缩目标本来已有DP下界。接下来优先考虑词表选择+序列决策的联合优化、数学操作符/命令原子目标、多指标Pareto与更多独立PDE文档，而不是把目前小幅压缩当最终贡献。',
        '没有训练下游模型；经验unigram bits/byte不是LM BPB，不报告推理提升。']
    (ROOT/'docs/v2_findings.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
