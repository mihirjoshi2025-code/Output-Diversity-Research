---
tags: [vocab, training-method]
aliases: [SFT, supervised fine-tuning, instruction tuning]
---
# Supervised Fine-Tuning (SFT)

**In one line:** teach the model by showing it thousands of example (question → ideal answer) pairs and training it to copy the answers.

**How it works, step by step:**
1. Collect examples: a prompt plus a good response. For OLMo 3 Think, most responses were written by *other, stronger* AI models ([[Teacher Model|teacher models]]) such as DeepSeek R1 and QwQ-32B.
2. Show the model the prompt and ask it to predict the response one [[Token|token]] at a time.
3. Every time its prediction differs from the real next token, measure the error ([[Loss|loss]]) and nudge the [[Parameters|weights]] so that token becomes more likely next time.
4. Repeat over millions of examples (OLMo: ~2.27M examples, 2 [[Epoch|epochs]]).

**Analogy:** a student copying worked solutions from the answer key until they can produce that kind of answer themselves.

**Key property:** SFT learns the *whole distribution* of the examples. If 3% of the story examples are Core stories, a perfect imitator would write roughly 3% Core stories. It has no reason to *exaggerate* a pattern.

**Connection to the paper:** Core share in the SFT data = **3.0%** (Table 2). If the SFT checkpoint already writes, say, 50% Core stories, imitation alone can't explain it, and the habit may come from the [[Base Model|base model]] or from how SFT generalizes.

**Details:** [[OLMo 3 Think - SFT]]
**Related:** [[Direct Preference Optimization (DPO)]], [[Synthetic Data]], [[Teacher Model]]
