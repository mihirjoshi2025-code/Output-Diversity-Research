---
tags: [vocab]
aliases: [teacher model, teacher, distillation, teacher models]
---
# Teacher Model (Distillation)

**In one line:** a stronger model whose answers are used to train a smaller one. Training a student on a teacher's outputs is called **distillation**.

**Plain explanation:** Writing millions of training answers by hand is too expensive, so labs have big models write them. The student picks up the teacher's knowledge *and* its style and quirks.

**OLMo 3 Think's teachers by stage:**
| Stage | Teacher for chat and stories |
|---|---|
| [[Supervised Fine-Tuning (SFT)\|SFT]] | DeepSeek R1 (chat), QwQ-32B (math, code) |
| [[Direct Preference Optimization (DPO)\|DPO]] | Qwen3-32B (chosen) vs Qwen3-0.6B (rejected) |
| [[RLVR]] | Qwen3-32B as [[LM Judge]] |

**Why it matters for us:** OLMo's story style may be inherited. Models from *different* labs share Core habits (our scorer on `core_rates.json`: Claude Haiku 4.5 91.4%, Gemini 3.1 Flash Lite 99.0%, GPT-5.4 mini 89.6%, OLMo 73.1%), and models learning from each other's outputs is one possible way the habits spread.

**Related:** [[Synthetic Data]], [[Delta Learning]]
