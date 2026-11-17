---
tags: [vocab]
aliases: [model merging, model soup, soup, merge, mergekit]
---
# Model Merging ("Model Soup")

**In one line:** combining two or more trained models into one by averaging their [[Parameters|weights]].

**Plain explanation:** Ai2 trained SFT twice with different [[Learning Rate|learning rates]] and averaged the two results (with a tool called mergekit). Surprisingly, the averaged model is often better than either one.

**For us:** the `Olmo-3-7B-Think-SFT` checkpoint is this merged model, not a single training run.

**Related:** [[Supervised Fine-Tuning (SFT)]], [[Checkpoint]]
