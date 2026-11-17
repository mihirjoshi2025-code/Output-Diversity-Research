---
tags: [vocab]
aliases: [preference pair, preference pairs, chosen, rejected, chosen and rejected, preference data]
---
# Preference Pair (chosen / rejected)

**In one line:** one prompt plus two answers, one labelled better (**chosen**) and one worse (**rejected**). This is the training data for [[Direct Preference Optimization (DPO)|DPO]].

**Example:**
- Prompt: "Write a short story about the sea."
- Chosen: a polished story from Qwen3-32B.
- Rejected: a weaker story from Qwen3-0.6B.

**Who decides "better"?** Usually humans or an AI judge. In OLMo 3 *nobody* judged: the bigger model's answer was assumed to be better ([[Delta Learning]]).

**In our project:** `training_stories/data/dolci-think-dpo-7b.jsonl` holds story-like pairs. A planned test is to run the Core scorer on the chosen side and the rejected side **separately**. If chosen stories are more Core than rejected ones, DPO was literally rewarding Core-ness.

**Related:** [[Direct Preference Optimization (DPO)]], [[Delta Learning]]
