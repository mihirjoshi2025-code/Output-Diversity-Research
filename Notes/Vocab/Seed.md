---
tags: [vocab, sampling]
aliases: [seed, random seed]
---
# Seed

**In one line:** the starting number for the model's dice.

**Plain explanation:** The model picks each word by rolling weighted dice ([[Temperature|sampling]]). The seed decides which rolls come up. **Same prompt + same seed + same settings = same story.** Different seed = a different story.

**Why it matters for us:** our script gives every story its own seed (`prompt × 100,000 + story number`), so no two stories are copies, and story #17 of a prompt gets the same seed on every checkpoint. Not a perfect guarantee: a different llama.cpp version can still change the output slightly.

**Related:** [[Temperature]], [[Top-p]]
