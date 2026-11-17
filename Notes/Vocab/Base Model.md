---
tags: [vocab]
aliases: [base model, base]
---
# Base Model

**In one line:** the model straight out of [[Pretraining|pretraining]]. Smart autocomplete, not yet a chat assistant.

**Plain explanation:** A base model has absorbed a huge amount of text but has no idea it is supposed to be a helpful assistant. It just continues whatever text you give it. [[Post-training]] (SFT → DPO → RL) turns it into something that answers questions and follows instructions.

**In our project:** the starting point for OLMo 3 Think's post-training. Mentor question #4 asks whether to include it as a fourth comparison point. It would show whether Core habits exist even *before* post-training.

**Related:** [[Pretraining]], [[Post-training]], [[Supervised Fine-Tuning (SFT)]]
