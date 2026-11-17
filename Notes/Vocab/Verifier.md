---
tags: [vocab]
aliases: [verifier, verifiers]
---
# Verifier

**In one line:** a small program that checks whether an answer is correct, with no opinions involved.

**Examples in OLMo 3:**
- **Math:** SymPy (a math library) checks whether the model's answer equals the reference answer.
- **Code:** runs test cases.
- **Instruction following:** a function checks a rule like "no commas" or "exactly 3 bullet points."

**Contrast:** an [[LM Judge]] gives an *opinion*; a verifier gives a *fact*. Stories can only get the opinion kind.

**Related:** [[RLVR]], [[Reward]], [[Instruction Following]]
