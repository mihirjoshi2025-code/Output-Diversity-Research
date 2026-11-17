---
tags: [vocab]
aliases: [entropy]
---
# Entropy

**In one line:** a measure of how spread out (unpredictable) a model's choices are. High entropy = many options; low entropy = it nearly always picks the same thing.

**Example:** picking the next word after "Once upon a time, there was a ___":
- **High entropy:** "girl" 10%, "dragon" 8%, "king" 7%, "baker" 6%, ... many possibilities.
- **Low entropy:** "lighthouse keeper" 80%.

**In the OLMo 3 report:** the DPO checkpoint has **lower entropy than SFT**. This is the report's only direct comment on diversity, and it fits the idea that DPO narrows outputs.

**Related:** [[Mode Collapse]], [[Temperature]], [[Pass@k]]
