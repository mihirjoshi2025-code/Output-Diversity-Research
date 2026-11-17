---
tags: [vocab]
aliases: [fine-tuning, fine-tune, finetuning, fine-tuned]
---
# Fine-tuning

**In one line:** extra training on a smaller, targeted dataset to adjust an already-trained model.

**Plain explanation:** Instead of training from scratch, you take a model that already knows language and train it a bit more on specific data, for example example conversations. The [[Parameters|weights]] move only slightly, but behaviour can change a lot. SFT, DPO and RL are all kinds of fine-tuning. They differ in *what signal* the model learns from:

| Method | Signal |
|---|---|
| [[Supervised Fine-Tuning (SFT)\|SFT]] | "Here is the right answer, copy it." |
| [[Direct Preference Optimization (DPO)\|DPO]] | "Answer A is better than answer B." |
| [[Reinforcement Learning\|RL]] | "You wrote this; it got a score of 0.8." |

**Related:** [[Post-training]], [[Loss]]
