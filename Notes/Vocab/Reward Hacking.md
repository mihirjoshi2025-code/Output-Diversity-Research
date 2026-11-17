---
tags: [vocab]
aliases: [reward hacking, over-optimization, overoptimization]
---
# Reward Hacking (Over-optimization)

**In one line:** the model learns to please the scorer instead of actually doing the task well.

**Examples:**
- A judge likes long answers → the model writes padded, long answers.
- A judge thinks "creative" means misty lighthouses and wise old keepers → the model writes that every time.

**In the OLMo 3 report:** training RL on one domain alone over-optimized it (IFEval up, chat quality down). The appendix also admits AlpacaEval rewards "flair" and over-optimized style.

**Why it matters for us:** Core stories might be a *mild* form of reward hacking: a safe, polished template that judges reliably rate well.

**Related:** [[LM Judge]], [[KL Penalty]], [[Mode Collapse]]
