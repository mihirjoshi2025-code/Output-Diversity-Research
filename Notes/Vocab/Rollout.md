---
tags: [vocab]
aliases: [rollout, rollouts, sample, samples, generation]
---
# Rollout

**In one line:** one answer the model writes on its own during [[Reinforcement Learning|RL]], which then gets scored.

OLMo's RL rollouts averaged **over 10,000 [[Token|tokens]]** (mostly reasoning), and were sampled at [[Temperature|temperature]] 1.0. That is much longer than our 4,096-token cap. One reason to check `finish_reason` in our runs.

**Related:** [[GRPO]], [[Reward]]
