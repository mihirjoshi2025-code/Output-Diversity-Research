---
tags: [vocab]
aliases: [KL penalty, KL divergence, KL, KL regularization]
---
# KL Penalty

**In one line:** a "leash" in training that penalizes the model for drifting too far from its starting point.

**Plain explanation:** KL divergence is a measure of how different two models' word probabilities are. Adding it as a penalty during RL means: "get more reward, but don't become a completely different model." Without the leash, a model can chase the reward into weird, narrow behaviour.

**OLMo 3's RL uses no KL penalty.** Nothing explicitly stops the model from drifting toward whatever the judge likes. That is a plausible reason a pattern could grow sharply during RL.

**Related:** [[Beta (DPO)]], [[Policy]], [[Reward Hacking]], [[Mode Collapse]]
