---
tags: [vocab]
aliases: [instruction following, IF, precise IF, precise instruction following]
---
# Instruction Following (IF)

**In one line:** how well a model obeys specific, checkable rules in a prompt, e.g. "answer in exactly 3 paragraphs" or "don't use commas."

OLMo 3 trains this in SFT (examples) and RL ([[Verifier|verifier]] rewards). RL gave the biggest gain here (IFEval 75.9 → 88.2).

**For us:** these rewards could shape a story's *form* (length, structure) but not names or settings. They're unlikely to directly cause Core stories.

**Related:** [[Benchmark]], [[RLVR]]
