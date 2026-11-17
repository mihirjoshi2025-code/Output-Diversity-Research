---
tags: [vocab]
aliases: [beta, β, DPO beta]
---
# β (Beta) in DPO

**In one line:** a setting that controls how strongly [[Direct Preference Optimization (DPO)|DPO]] keeps the model close to where it started.

**Plain explanation:** DPO pushes the model toward chosen answers, but it also measures everything *relative to the reference model* (the SFT checkpoint). β sets how much that comparison matters. Higher β roughly means "be more conservative, don't drift far." It plays the same role as a [[KL Penalty|KL penalty]] in RL.

**OLMo 3:** β = 5. This is much larger than the 0.1 often used in the original DPO paper, but OLMo uses **length-normalized** DPO, which changes the scale, so the numbers can't be compared directly.

**Related:** [[KL Penalty]], [[Policy]]
