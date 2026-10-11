# 全部正式实验的数值附录

由 scripts/all_model_review.py 从保存的逐公式记录独立核对；没有重训或重新编码。
token/公式、跨树比例越低越好；命令完整率、编码速度越高越好。速度是原实验记录，未在本次重测。
训练秒的统计口径跨版本不同，因此不做跨版本训练时间排名。参考模型、控制组与正式方法用 role 区分。
重复 seeds 不等于独立发现；本表是描述性排序，不自动表示相邻名次具有统计显著差异。

## pilot01

原始逐公式记录 10,556；方法/种子/场景记录 26。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| tiktoken_cl100k_base | pretrained_reference | 0 | 12.3005 | — | — | 10.1051 |
| tiktoken_o200k_base | pretrained_reference | 0 | 12.3103 | — | — | 13.9771 |
| qwen2.5_official | pretrained_reference | 0 | 12.3448 | — | — | 2.7005 |
| byte_bpe | baseline | 17,29,43 | 14.3768 | — | — | 1.7161 |
| hf_byte_bpe | baseline | 0 | 14.3768 | — | — | 6.5907 |
| byte_bpe_rl_fit_control | control | 17,29,43 | 14.4261 | — | — | 1.7047 |
| rl_bpe | proposed | 17,29,43 | 14.4787 | — | — | 1.6975 |
| tiktoken_gpt2 | pretrained_reference | 0 | 14.7882 | — | — | 9.1571 |
| ast_bpe | proposed | 17,29,43 | 17.5369 | — | — | 0.8095 |
| ast_bpe_rl_fit_control | control | 17,29,43 | 17.5550 | — | — | 0.8103 |
| sentencepiece_unigram | baseline | 0 | 17.6059 | — | — | 16.1154 |
| sentencepiece_bpe | baseline | 0 | 17.6305 | — | — | 11.2031 |
| rl_ast_bpe | proposed | 17,29,43 | 17.6437 | — | — | 0.8292 |
| utf8_bytes | fallback_reference | 0 | 24.6207 | — | — | 143.9845 |

## sensitivity512

原始逐公式记录 10,556；方法/种子/场景记录 26。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| tiktoken_cl100k_base | pretrained_reference | 0 | 12.3005 | — | — | 9.9414 |
| tiktoken_o200k_base | pretrained_reference | 0 | 12.3103 | — | — | 13.6047 |
| qwen2.5_official | pretrained_reference | 0 | 12.3448 | — | — | 2.6794 |
| rl_bpe | proposed | 17,29,43 | 12.5320 | — | — | 1.0728 |
| byte_bpe_rl_fit_control | control | 17,29,43 | 12.5706 | — | — | 1.0822 |
| byte_bpe | baseline | 17,29,43 | 12.6158 | — | — | 1.0784 |
| hf_byte_bpe | baseline | 0 | 12.6158 | — | — | 6.6837 |
| sentencepiece_unigram | baseline | 0 | 14.2044 | — | — | 17.0434 |
| sentencepiece_bpe | baseline | 0 | 14.3571 | — | — | 11.2400 |
| tiktoken_gpt2 | pretrained_reference | 0 | 14.7882 | — | — | 10.0913 |
| ast_bpe_rl_fit_control | control | 17,29,43 | 16.8144 | — | — | 0.7022 |
| ast_bpe | proposed | 17,29,43 | 16.8177 | — | — | 0.7091 |
| rl_ast_bpe | proposed | 17,29,43 | 16.8432 | — | — | 0.7018 |
| utf8_bytes | fallback_reference | 0 | 24.6207 | — | — | 133.2494 |

## v2_release512

原始逐公式记录 91,057；方法/种子/场景记录 23。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| byte_shortest_path | control | 0 | 18.6052 | 55.81% | 14.20% | 0.1521 |
| rl_bpe_v2 | proposed | 17,29,43 | 18.6763 | 53.10% | 14.29% | 0.0963 |
| byte_policy_no_rl | control | 0 | 18.7651 | 49.74% | 15.23% | 0.0958 |
| byte_bpe | baseline | 0 | 18.8636 | 47.49% | 15.41% | 2.7058 |
| hf_byte_bpe | baseline | 0 | 18.8636 | 47.49% | 15.41% | 6.8595 |
| tiktoken_o200k_base | pretrained_reference | 0 | 19.0336 | 0.00% | 21.82% | 12.0926 |
| tiktoken_cl100k_base | pretrained_reference | 0 | 19.1715 | 0.00% | 21.57% | 8.8522 |
| rl_bpe_v1 | proposed | 17 | 19.2354 | 24.60% | 18.38% | 1.3661 |
| qwen2.5_official | pretrained_reference | 0 | 19.3562 | 0.00% | 21.35% | 2.5274 |
| ast_shortest_path | control | 0 | 19.9366 | 76.22% | 0.00% | 0.1180 |
| ast_policy_no_rl | control | 0 | 19.9656 | 76.22% | 0.00% | 0.0817 |
| rl_ast_bpe_v2 | proposed | 17,29,43 | 19.9667 | 76.22% | 0.00% | 0.0812 |
| ast_bpe_v2 | proposed | 0 | 20.1018 | 75.59% | 0.00% | 0.5447 |
| sentencepiece_unigram | baseline | 0 | 21.5605 | 0.00% | 13.07% | 17.1575 |
| sentencepiece_bpe | baseline | 0 | 21.9836 | 0.00% | 15.63% | 11.6222 |
| tiktoken_gpt2 | pretrained_reference | 0 | 22.6527 | 0.00% | 14.20% | 9.5628 |
| ast_bpe_v1 | proposed | 17 | 32.6042 | 15.89% | 0.00% | 1.4550 |
| rl_ast_bpe_v1 | proposed | 17 | 32.6135 | 15.89% | 0.00% | 1.4693 |
| utf8_bytes | fallback_reference | 0 | 35.8159 | 0.00% | 0.00% | 91.5263 |

## v2_release1024

原始逐公式记录 91,057；方法/种子/场景记录 23。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| byte_shortest_path | control | 0 | 14.7239 | 89.22% | 17.71% | 0.1421 |
| rl_bpe_v2 | proposed | 17,29,43 | 14.9190 | 81.44% | 18.64% | 0.0933 |
| byte_policy_no_rl | control | 0 | 14.9836 | 78.11% | 18.78% | 0.0930 |
| byte_bpe | baseline | 0 | 15.0573 | 77.86% | 19.02% | 2.2794 |
| hf_byte_bpe | baseline | 0 | 15.0573 | 77.86% | 19.02% | 7.7039 |
| rl_bpe_v1 | proposed | 17 | 15.2983 | 51.67% | 22.94% | 0.6023 |
| ast_shortest_path | control | 0 | 15.6085 | 92.68% | 0.00% | 0.1102 |
| ast_policy_no_rl | control | 0 | 15.6388 | 92.69% | 0.00% | 0.0780 |
| rl_ast_bpe_v2 | proposed | 17,29,43 | 15.6443 | 92.69% | 0.00% | 0.0778 |
| ast_bpe_v2 | proposed | 0 | 15.8032 | 92.05% | 0.00% | 0.4682 |
| sentencepiece_bpe | baseline | 0 | 17.8831 | 0.00% | 22.47% | 10.8102 |
| sentencepiece_unigram | baseline | 0 | 18.4084 | 0.00% | 18.10% | 18.5615 |
| tiktoken_o200k_base | pretrained_reference | 0 | 19.0336 | 0.00% | 21.82% | 13.5097 |
| tiktoken_cl100k_base | pretrained_reference | 0 | 19.1715 | 0.00% | 21.57% | 9.3258 |
| qwen2.5_official | pretrained_reference | 0 | 19.3562 | 0.00% | 21.35% | 2.7109 |
| tiktoken_gpt2 | pretrained_reference | 0 | 22.6527 | 0.00% | 14.20% | 9.3896 |
| ast_bpe_v1 | proposed | 17 | 31.8783 | 17.90% | 0.00% | 0.9384 |
| rl_ast_bpe_v1 | proposed | 17 | 31.8954 | 17.88% | 0.00% | 0.9436 |
| utf8_bytes | fallback_reference | 0 | 35.8159 | 0.00% | 0.00% | 78.3939 |

## v3_512_release

原始逐公式记录 71,262；方法/种子/场景记录 18。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| atomic_byte_shortest_control | control | 0 | 17.0482 | 86.69% | 14.64% | 1.1831 |
| rl_bpe_v3 | proposed | 17,29,43 | 17.2916 | 86.83% | 13.17% | 1.1718 |
| ast_bpe_v3 | proposed | 0 | 18.5090 | 96.50% | 0.00% | 1.1790 |
| rl_ast_bpe_v3 | proposed | 17,29,43 | 18.5289 | 96.50% | 0.00% | 1.1764 |
| rl_bpe_v2 | proposed | 17,29,43 | 18.6763 | 53.10% | 14.29% | 0.0969 |
| rl_bpe_v2_native_exact | engineering_control | 17,29,43 | 18.6763 | 53.10% | 14.29% | 1.1676 |
| rl_ast_bpe_v2 | proposed | 17,29,43 | 19.9667 | 76.22% | 0.00% | 0.0822 |
| ast_bpe_v2 | proposed | 0 | 20.1018 | 75.59% | 0.00% | 0.5391 |

## v3_1024_release

原始逐公式记录 71,262；方法/种子/场景记录 18。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| atomic_byte_shortest_control | control | 0 | 13.9172 | 96.15% | 19.01% | 1.1573 |
| rl_bpe_v3 | proposed | 17,29,43 | 13.9361 | 96.50% | 18.18% | 1.1479 |
| rl_bpe_v2 | proposed | 17,29,43 | 14.9190 | 81.44% | 18.64% | 0.0919 |
| rl_bpe_v2_native_exact | engineering_control | 17,29,43 | 14.9190 | 81.44% | 18.64% | 1.1459 |
| ast_bpe_v3 | proposed | 0 | 14.9752 | 96.50% | 0.00% | 1.1707 |
| rl_ast_bpe_v3 | proposed | 17,29,43 | 15.0062 | 96.50% | 0.00% | 1.1527 |
| rl_ast_bpe_v2 | proposed | 17,29,43 | 15.6443 | 92.69% | 0.00% | 0.0775 |
| ast_bpe_v2 | proposed | 0 | 15.8032 | 92.05% | 0.00% | 0.4678 |

## v4_512

原始逐公式记录 55,426；方法/种子/场景记录 14。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| rl_v3 | proposed | 17,29,43 | 17.2916 | 86.83% | 13.17% | 1.1578 |
| ast_v3 | proposed | 0 | 18.5090 | 96.50% | 0.00% | 1.1847 |
| ast_v4 | proposed | 0 | 19.4264 | 99.52% | 0.00% | 1.0847 |
| masked_shortest_control | control | 0 | 19.4264 | 99.52% | 0.00% | 1.1071 |
| rl_v4 | proposed | 17,29,43 | 19.4269 | 99.52% | 0.00% | 1.0980 |
| v3_mask_only_control | control | 17,29,43 | 20.0418 | 86.83% | 0.00% | 1.0819 |
| masked_matched_initial_policy | control | 0 | 22.9520 | 99.52% | 0.00% | 1.0926 |
| masked_zero_policy | control | 0 | 22.9520 | 99.52% | 0.00% | 1.0951 |

## v4_1024

原始逐公式记录 55,426；方法/种子/场景记录 14。

场景：historical/test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| rl_v3 | proposed | 17,29,43 | 13.9361 | 96.50% | 18.18% | 1.1484 |
| ast_v4 | proposed | 0 | 14.9227 | 99.52% | 0.00% | 1.0841 |
| masked_shortest_control | control | 0 | 14.9227 | 99.52% | 0.00% | 1.0872 |
| masked_matched_initial_policy | control | 0 | 14.9437 | 99.52% | 0.00% | 1.0793 |
| rl_v4 | proposed | 17,29,43 | 14.9462 | 99.52% | 0.00% | 1.0718 |
| ast_v3 | proposed | 0 | 14.9752 | 96.50% | 0.00% | 1.1633 |
| v3_mask_only_control | control | 17,29,43 | 17.1295 | 96.50% | 0.00% | 1.0745 |
| masked_zero_policy | control | 0 | 19.3428 | 99.52% | 0.00% | 1.0696 |

## v5_1024

原始逐公式记录 106,797；方法/种子/场景记录 54。

场景：broad/fresh_paper_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| reward_greedy | control | 17 | 21.5458 | 89.89% | 25.68% | 8.8563 |
| rl_composite | proposed | 17,29,43 | 21.7221 | 87.75% | 26.22% | 8.6835 |
| frequency | control | 17 | 22.1778 | 78.26% | 25.72% | 9.0044 |
| hf_bpe_matched_fit | baseline | 0 | 22.1778 | 78.26% | 25.72% | 8.5901 |
| hf_bpe_fulltrain | baseline | 0 | 22.3751 | 76.26% | 25.92% | 8.6436 |
| rl_length_only | control | 17 | 22.7858 | 64.32% | 28.26% | 8.7285 |
| random | control | 17 | 23.5623 | 48.27% | 30.09% | 8.8256 |
| sp_bpe | baseline | 0 | 27.9790 | 0.00% | 25.23% | 11.2085 |
| sp_unigram | baseline | 0 | 29.0197 | 0.00% | 19.14% | 22.0904 |

场景：broad/synthetic_stress_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| reward_greedy | control | 17 | 102.1443 | 91.90% | 36.09% | 7.5927 |
| rl_composite | proposed | 17,29,43 | 102.9149 | 90.77% | 35.98% | 8.6275 |
| hf_bpe_fulltrain | baseline | 0 | 102.9221 | 84.23% | 36.20% | 8.6743 |
| rl_length_only | control | 17 | 103.4719 | 77.91% | 36.89% | 8.7007 |
| frequency | control | 17 | 103.5440 | 85.56% | 35.85% | 8.9858 |
| hf_bpe_matched_fit | baseline | 0 | 103.5440 | 85.56% | 35.85% | 8.8387 |
| random | control | 17 | 105.7100 | 62.59% | 37.41% | 8.7360 |
| sp_bpe | baseline | 0 | 118.1515 | 0.00% | 29.40% | 11.5111 |
| sp_unigram | baseline | 0 | 121.3261 | 0.00% | 26.47% | 36.5534 |

场景：prior_data_reference/extreme_domain_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| legacy_v3_rl | prior_data_reference | 17 | 13.9334 | 96.95% | 15.27% | 1.0318 |
| legacy_v4_masked_rl | prior_data_reference | 17 | 15.0966 | 100.00% | 0.00% | 0.9690 |

场景：prior_data_reference/fresh_paper_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| legacy_v3_rl | prior_data_reference | 17 | 21.1273 | 98.46% | 22.65% | 1.4008 |
| legacy_v4_masked_rl | prior_data_reference | 17 | 23.2439 | 99.96% | 0.00% | 1.3147 |

场景：reference/extreme_domain_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| tiktoken_cl100k_base | pretrained_reference | 0 | 18.0086 | 0.00% | 22.26% | 9.0759 |
| tiktoken_o200k_base | pretrained_reference | 0 | 18.0648 | 0.00% | 22.35% | 12.8572 |

场景：reference/fresh_paper_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| tiktoken_o200k_base | pretrained_reference | 0 | 30.7191 | 0.00% | 22.47% | 13.2376 |
| tiktoken_cl100k_base | pretrained_reference | 0 | 31.1159 | 0.00% | 22.02% | 10.0638 |

场景：reference/synthetic_stress_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| tiktoken_o200k_base | pretrained_reference | 0 | 112.8874 | 0.00% | 41.93% | 19.3702 |
| tiktoken_cl100k_base | pretrained_reference | 0 | 121.4358 | 0.00% | 38.85% | 13.0912 |

场景：specialized/extreme_domain_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| reward_greedy | control | 17 | 15.5012 | 74.69% | 15.38% | 7.3608 |
| random | control | 17 | 15.5685 | 73.44% | 14.93% | 6.7616 |
| rl_composite | proposed | 17,29,43 | 15.5892 | 71.86% | 15.03% | 7.1121 |
| hf_bpe_fulltrain | baseline | 0 | 15.7139 | 58.63% | 16.20% | 7.3874 |
| rl_length_only | control | 17 | 15.7274 | 52.24% | 17.05% | 7.0912 |
| frequency | control | 17 | 16.0043 | 47.68% | 18.99% | 7.1701 |
| hf_bpe_matched_fit | baseline | 0 | 16.0043 | 47.68% | 18.99% | 7.3393 |
| sp_unigram | baseline | 0 | 16.7604 | 0.00% | 18.47% | 17.9709 |
| sp_bpe | baseline | 0 | 17.0795 | 0.00% | 21.32% | 10.7923 |

场景：specialized/synthetic_stress_test。

| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |
|---|---|---|---:|---:|---:|---:|
| hf_bpe_fulltrain | baseline | 0 | 108.0361 | 83.34% | 31.05% | 8.8503 |
| frequency | control | 17 | 109.1039 | 79.38% | 30.92% | 8.6366 |
| hf_bpe_matched_fit | baseline | 0 | 109.1039 | 79.38% | 30.92% | 8.6965 |
| rl_length_only | control | 17 | 109.1241 | 78.97% | 30.68% | 8.9398 |
| rl_composite | proposed | 17,29,43 | 109.2973 | 88.53% | 29.86% | 8.8974 |
| reward_greedy | control | 17 | 112.7316 | 88.16% | 29.31% | 9.0393 |
| random | control | 17 | 112.9856 | 86.21% | 29.54% | 8.8984 |
| sp_bpe | baseline | 0 | 118.5123 | 0.00% | 29.05% | 11.6424 |
| sp_unigram | baseline | 0 | 119.8456 | 0.00% | 27.99% | 34.2851 |
