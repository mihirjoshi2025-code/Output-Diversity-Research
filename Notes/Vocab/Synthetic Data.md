---
tags: [vocab]
aliases: [synthetic data, synthetic]
---
# Synthetic Data

**In one line:** training data written by AI models instead of people.

**Plain explanation:** Most of OLMo 3's [[Post-training|post-training]] data is synthetic. Humans (or real users, as in [[WildChat]]) supplied the *prompts*; other models ([[Teacher Model|teachers]]) wrote the *answers*. It is cheap and scalable, but it passes along the teacher's habits.

**Why it matters for us:** the stories in `training_stories/` were mostly written by DeepSeek R1 and Qwen3 models, not humans. A Core story in the training data is one an AI already wrote, which may be part of how the pattern spreads.

**Related:** [[Teacher Model]], [[Dolci]]
