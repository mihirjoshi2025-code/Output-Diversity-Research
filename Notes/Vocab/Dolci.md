---
tags: [vocab]
aliases: [Dolci, Dolci-Think-SFT, Dolci-Think-DPO, Dolci-Think-RL]
---
# Dolci

**In one line:** the name of OLMo 3's [[Post-training|post-training]] datasets.

| Dataset | Stage | Core share (paper Table 2) |
|---|---|---|
| Dolci-Think-SFT | [[Supervised Fine-Tuning (SFT)\|SFT]] | 3.0% |
| Dolci-Think-DPO | [[Direct Preference Optimization (DPO)\|DPO]] | 8.0% |
| Dolci-Think-RL | [[RLVR]] | 10.9% |

`training_stories/data/` holds the story-like extracts from these. Text is in field `text`. Some aren't stories, and some are explicit.

**Related:** [[WildChat]], [[Synthetic Data]]
