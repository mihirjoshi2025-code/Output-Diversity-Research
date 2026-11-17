---
tags: [vocab]
aliases: [post-training, posttraining, post-trained]
---
# Post-training

**In one line:** everything done to a [[Base Model|base model]] after [[Pretraining|pretraining]] to make it a useful, well-behaved assistant.

**Plain explanation:** Pretraining teaches the model *language*. Post-training teaches it *behaviour*: answer the question, follow the format, be safe, show its reasoning. It uses far less data than pretraining but changes the model's personality a lot. For OLMo 3 Think it has three steps:

1. **[[Supervised Fine-Tuning (SFT)|SFT]]**: copy good examples (imitation).
2. **[[Direct Preference Optimization (DPO)|DPO]]**: learn "this answer is better than that one" (preference).
3. **[[RLVR]]**: try things, get scored, do more of what scores well (reward).

**Why it matters for us:** the paper's question is whether post-training *narrows* what the model writes. Research like Kirk et al. 2024 found preference training and RL reduce output variety ([[Mode Collapse]]). We test which of the three steps does it for Core stories.

**Related:** [[Fine-tuning]], [[Checkpoint]], [[Dolci]]
