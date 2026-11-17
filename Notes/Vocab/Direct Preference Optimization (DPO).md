---
tags: [vocab, training-method]
aliases: [DPO, direct preference optimization, preference optimization, preference tuning]
---
# Direct Preference Optimization (DPO)

**In one line:** teach the model by showing it two answers to the same prompt, one marked better ("chosen"), one worse ("rejected"), and training it to prefer the better one.

**How it works, step by step:**
1. Collect [[Preference Pair|preference pairs]]: (prompt, chosen answer, rejected answer).
2. Ask the model: how likely would *you* be to write the chosen answer? The rejected one?
3. Nudge the [[Parameters|weights]] so the chosen answer becomes more likely **relative to** the rejected one, compared with where the model started (the *reference model*, here the SFT checkpoint).
4. A setting called [[Beta (DPO)|β (beta)]] controls how far the model may drift from the reference.

**Analogy:** a writing teacher who never writes an example essay, but repeatedly holds up two essays and says "this one is better." You learn to lean toward whatever the "better" pile has in common.

**Why "direct":** the older method ([[RLHF]]) first trained a separate "reward model" to score answers, then used [[Reinforcement Learning|RL]]. DPO (Rafailov et al. 2023) skips both and learns from the pairs directly with one simple formula. It is cheaper and more stable.

**Key property, and why it matters to us:** DPO learns from the **difference** between chosen and rejected. It doesn't simply copy the chosen answers. Any feature that shows up more in chosen answers than in rejected ones gets pushed up, *even if that feature is rare overall*. So DPO can **amplify** a pattern far beyond its share of the data.

**Connection to the paper:** Core share in the DPO data = **8.0%** (Table 2). If the DPO checkpoint's Core rate jumps far above 8%, that points to amplification. That is the setup for Question 3. In OLMo 3, chosen = Qwen3-32B, rejected = Qwen3-0.6B ([[Delta Learning]]), so whatever story habits Qwen3-32B has more than Qwen3-0.6B get rewarded.

**Details:** [[OLMo 3 Think - DPO]]
**Related:** [[Preference Pair]], [[Delta Learning]], [[Beta (DPO)]], [[KL Penalty]], [[Mode Collapse]]
