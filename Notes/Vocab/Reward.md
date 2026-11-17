---
tags: [vocab]
aliases: [reward, rewards, reward signal, reward model]
---
# Reward

**In one line:** the score an answer gets during [[Reinforcement Learning|RL]]. The model is trained to get higher scores.

**Plain explanation:** A number, usually between 0 and 1, saying how good an answer was. The model never sees *why* it got the score, only the number. So it learns whatever raises the number, even accidental things the scorer happens to like (see [[Reward Hacking]]).

**Where rewards come from in OLMo 3:**
- [[Verifier|Verifiers]] for math, code and instruction following (right/wrong).
- An [[LM Judge]] for chat and stories (a 1-10 opinion, rescaled to 0-1).

A **reward model** is a model trained specifically to output reward scores (used in [[RLHF]]). OLMo 3 instead uses an off-the-shelf model (Qwen3-32B) prompted to act as a judge.

**Related:** [[RLVR]], [[Advantage]], [[Reward Hacking]]
