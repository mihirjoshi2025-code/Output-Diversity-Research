---
tags: [vocab, training-method]
aliases: [RLHF, reinforcement learning from human feedback]
---
# RLHF (Reinforcement Learning from Human Feedback)

**In one line:** the "classic" way to align chatbots: humans compare answers, a model learns their taste, and RL optimizes for that taste.

**Steps (as in InstructGPT, Ouyang et al. 2022):**
1. [[Supervised Fine-Tuning (SFT)|SFT]] on human-written examples.
2. Humans rank several model answers. A separate **reward model** is trained to predict those rankings.
3. [[Reinforcement Learning|RL]] (an algorithm called PPO) trains the chatbot to get high reward-model scores, with a [[KL Penalty|KL penalty]] so it doesn't drift too far.

**Why it matters for us:** OLMo 3 doesn't use classic RLHF, but its stages are descendants of it. DPO replaces steps 2-3 with one simpler step, and the [[LM Judge]] replaces the human-trained reward model. **Kirk et al. 2024** (on the reading list) found RLHF makes outputs *less diverse* than SFT, which is the same kind of effect we are looking for.

**Related:** [[Direct Preference Optimization (DPO)]], [[Mode Collapse]], [[Reward]]
