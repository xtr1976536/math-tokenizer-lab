# arXiv与其他公开公式来源 v2

官方规则核对：[API Terms](https://info.arxiv.org/help/api/tou.html)、[Permissions and Reuse](https://info.arxiv.org/help/license/reuse.html)。单连接、请求至少4秒间隔，本地缓存、限量OAI元数据窗口、禁止绕过403/429；不执行TeX、不递归爬网页。官方通常允许每3秒1次，当前实现更保守。

源码端点 `https://arxiv.org/src/<id><version>` 已实际试通；例如v2不存在而v1可下载时，按OAI版本选择存在的版本，不猜最新URL。OAI的from/until是记录更新时间，不是发表年，因此另外解析version日期，并仅接受最新版本也早于2020的稿件。下载后再检查对应版本abs页面的license，CC BY或CC0才允许进入可上传语料；nonexclusive/NC/ND/SA均未自动纳入。arXiv元数据可使用CC0不等于论文内容CC0。

当前98篇、每篇版本、日期、作者、title、categories、source URL、许可URL、抽象页、SHA-256、来源推定依据存于data/arxiv_sources.json。标题和许可元数据可发布；源码包仅缓存data/raw/arxiv，不上传。发布的公式保留逐条署名、license、文档位置和transformations，许可继承对应CC BY/CC0。

历史人工来源：按用户确定的2020年前论文规则分类，origin=human、origin_basis=pre2020_arxiv_historical_inference_user_approved；这是年代与学术作者依据，不是每条公式作者无机器辅助声明。版本必须早于2020，不能把早期首稿的2026修订版当历史人工源。

领域：大部分标签是论文主题域，不代表每个式子的严格数学类型。math.DS对应ode/dynamical-systems主题桶，包含动力系统里的辅助代数表达式；PDE桶使用显式偏导／空间算子筛选，其余辅助表达式回analysis。线性代数式单独以矩阵、det/rank/spec操作符规则标注。需进一步专家校对主题/公式类型双标签；现阶段不能宣称自动领域分类完全无误。

复杂度：两档仅描述表示层复杂度，不代表证明难度。score=.40*L/60+.20*C/6+.15*D/3+.15*S/18+.10*O/12，各比值上限2；score>=.65为complex。L为公式词法单元数，C为命令种类数，D为显式分组最大嵌套深度，S为符号种类数，O为运算符数量。保留各原始分量便于之后重新分档。规则在正式test评估前固定，没有用模型成绩选阈值。

固定来源复现 `scripts/fetch_v2_sources.py` 只按已核验registry重下载，不重新发现候选和换版本；首次采集由 `scripts/harvest_arxiv.py` 实现，审计record在data/arxiv_harvest_audit.json。数据变更须发布新release，不能覆盖当前实验的hash。

Fungrim：[官方仓库](https://github.com/fredrik-johansson/fungrim)，MIT，不是CC0。固定 `b7c3ca6e565e1058638cab6ba2bc811090296767`；用原始pygrim .latex()导出2764个Formula条目，筛选后按公式接受。未知计算机辅助来源保留unknown，语义S表达式与LaTeX并存；转换是官方渲染，不是LLM自由生成。代码和MIT许可证原样保留。

注意：保留公式本体是tokenizer目标。许多特殊函数公式成立条件在上下文里，不能把从上下文取出的等式当无条件数学定理；当前语料用途是分词，不是公式真值监督。公式输出不含自然语言，元数据另存完整来源便于追溯。
