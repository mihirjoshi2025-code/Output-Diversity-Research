---
tags: [vocab, training-method]
aliases: [RL, reinforcement learning]
---
# Reinforcement Learning (RL)

**In one line:** learning by trial and error. The model tries something, gets a score ([[Reward|reward]]), and shifts toward whatever scored well.

**How it works for language models:**
1. Give the model a prompt.
2. Let it write several answers on its own ([[Rollout|rollouts]]).
3. Score each answer with a [[Reward|reward]]: a checker ([[Verifier|verifier]]) or another AI acting as judge ([[LM Judge]]).
4. Make high-scoring answers more likely and low-scoring ones less likely.
5. Repeat for many steps.

**Analogy:** training a dog with treats. You never show the dog the trick; you reward it when it happens to do it right.

**How it differs from SFT and DPO:** in SFT and DPO the model learns from answers *someone else* wrote. In RL the model learns from *its own* answers. That means it can reinforce its own existing habits: if it already likes lighthouses and the judge doesn't object, lighthouse stories keep getting rewarded.

**The model being trained** is called the [[Policy|policy]] in RL jargon.

**Related:** [[RLVR]], [[RLHF]], [[GRPO]], [[Reward Hacking]], [[KL Penalty]]
