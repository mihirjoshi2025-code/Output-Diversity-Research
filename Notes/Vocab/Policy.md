---
tags: [vocab]
aliases: [policy, reference model, reference policy]
---
# Policy (and Reference Model)

**In one line:** RL jargon for "the model being trained." It decides (its *policy*) which word to write next.

**Reference model:** a frozen copy of the model from *before* this training stage, used as an anchor for comparison. In OLMo's DPO the reference is the SFT checkpoint. Methods like the [[KL Penalty]] and [[Beta (DPO)|DPO's β]] measure how far the policy has moved from the reference.

**Related:** [[Reinforcement Learning]], [[KL Penalty]]
