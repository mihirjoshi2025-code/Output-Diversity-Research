---
tags: [vocab]
aliases: [advantage, advantages]
---
# Advantage

**In one line:** how much better (or worse) one answer scored than expected, usually than the group average.

**Example:** 8 stories score [0.9, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.3]. The average is 0.6. The 0.9 story has advantage +0.3 (strongly encouraged). The 0.3 story has −0.3 (discouraged).

OLMo does **not** divide by the spread of scores (no "std normalization"). Small differences between scores therefore stay small instead of being blown up.

**Related:** [[GRPO]], [[Reward]]
