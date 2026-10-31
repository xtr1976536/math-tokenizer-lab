# 第三套实验归档边界

正式结果入口为 `study3_final_report_zh.md`、`comparison.json` 和 `case_mechanism.csv`；正式训练与搜索数据在 `experiments/study3_corpus_effect_v1/`，大小语料均完成。

以下材料作为历史证据随仓库保留，不作为正式结果：

- `superseded_50_shared_vocab/`：最终采用60容量、大小组独立词表之前的准备稿与检查记录，已失效。
- `checks_4rounds_before_additional_review.json`：第五轮补充复核之前的检查快照。
- `historical_files_before.json`：启动前保护已有文件的哈希快照；后续仓库说明文件有授权更新，不能再将全仓库文字不变作为交付检查条件。
- `interrupted_before_cycle1/`：小语料对流扩散 EqGPT 在第一轮开始前中断的输入绑定；其中 cycle_0.pt 与正式共享预训练权重相同，不重复发布，关系见本目录的 checkpoint_alias.json。
- `small_final_report_zh.md`：小语料刚完成、大语料尚未启动时的阶段报告，状态文字是当时记录；最终以配对总报告为准。

`check_study3_integration.py` 和 `audit_study3.py --mode safety` 是启动前检查，要求正式运行目录尚不存在，不能在结果已经归档后当作交付验收命令运行。数据审计仍可用 `--mode data` 重跑。

`run_study3.py --phase plan` 返回的是静态准备计划，其中 AWAITING_USER_REVIEW 不代表已归档结果的完成状态。完成情况由逐组 complete.json、最终核验报告与 comparison.json 判断。

发布时增加下载脚本会进入 code_digest 的通配符扫描，因此历史四轮检查中的代码哈希不等于发布后的通配符哈希。原运行的训练、搜索代码和原始绑定保持原样；不通过刷新历史审核哈希伪造重跑授权。若重新训练或搜索，应重新审查并建立当前检查记录，不能混入既有正式结果。
