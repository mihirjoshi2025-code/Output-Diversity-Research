---
tags: [vocab, sampling]
aliases: [temperature, temp, sampling, sampling settings]
---
# Temperature (Sampling)

**In one line:** a dial controlling how random the model's word choices are.

**Plain explanation:** At each step the model has probabilities for the next [[Token|token]]. **Sampling** means picking one at random according to those probabilities. Temperature reshapes them:
- **0:** always pick the top choice. Same output every time.
- **0.6** (OLMo report's evaluation setting): somewhat focused.
- **1.0:** the model's raw probabilities (OLMo's RL training used this).
- **>1:** flatter, wilder, more mistakes.

**Why it matters for us:** temperature directly changes how often Core stories appear. Low temperature means more of the model's favourite (likely Core). So **every checkpoint must use the same settings**. We still need the old run's exact settings from the mentor.

**Related:** [[Top-p]], [[Entropy]]
