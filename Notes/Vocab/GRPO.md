---
tags: [vocab, training-method]
aliases: [GRPO, OlmoRL, group relative policy optimization]
---
# GRPO (and OlmoRL)

**In one line:** an RL recipe where the model writes several answers to the same prompt and learns by comparing them *with each other*.

**How it works:**
1. Take a prompt. Have the model write a **group** of answers (OLMo: 8).
2. Score each one ([[Reward|reward]]).
3. An answer's [[Advantage|advantage]] = its score minus the group's average score. Above average → make it more likely. Below → less likely.
4. Update the model and repeat.

**Analogy:** grading on a curve. You don't need to know what a "perfect" story is; you just push toward the better half of what the model itself wrote.

**Why it matters for us:** if the judge slightly prefers the Core story in each group, the Core story is always "above average," and RL steadily makes it more likely, without ever needing a Core story in the training data.

**OlmoRL** is Ai2's version of GRPO with a few changes: drop groups where everyone scored the same (no signal), refill the batch ("active sampling"), no [[KL Penalty|KL penalty]], and some technical tweaks (clip-higher, token-level loss, truncated importance sampling) that keep training stable.

**Related:** [[Rollout]], [[Advantage]], [[RLVR]]
