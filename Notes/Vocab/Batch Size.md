---
tags: [vocab]
aliases: [batch size, batch, batches]
---
# Batch Size

**In one line:** how many examples the model looks at before making one update.

**Plain explanation:** Instead of updating after every single example (noisy), the model averages the signal over a batch, for example 128 DPO pairs, then takes one step. Bigger batches give smoother, more reliable updates.

In OLMo's RL: 64 prompts per batch × 8 answers each ([[GRPO|group size]] 8).

**Related:** [[Epoch]], [[Learning Rate]]
