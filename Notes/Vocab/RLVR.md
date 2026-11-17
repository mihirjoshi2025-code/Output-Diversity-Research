---
tags: [vocab, training-method]
aliases: [RLVR, reinforcement learning with verifiable rewards, verifiable rewards]
---
# RLVR (Reinforcement Learning with Verifiable Rewards)

**In one line:** [[Reinforcement Learning|RL]] where the reward comes from an automatic *checker*, not a human or a guess. Example: "is the math answer correct?"

**Plain explanation:** For some tasks you can check the answer mechanically:
- **Math:** compare the final number with the known answer.
- **Code:** run the tests.
- **Instruction following:** check "did it use exactly 3 paragraphs?" with a small program.

These are "verifiable" because nobody's opinion is involved. The model gets 1 for right, 0 for wrong, and learns to be right more often.

**The catch for stories:** you can't verify a story. OLMo 3 handles chat and creative prompts with an [[LM Judge]] (Qwen3-32B grading the answer from 1 to 10). So "RLVR" is a bit of a misnomer for our purposes: **story prompts get an *opinion*-based reward**, not a verifiable one.

**Connection to the paper:** Core share in the RL data = **10.9%** (Table 2), and the final model writes **73.1%** Core stories. Whatever is left of the jump after DPO happens here.

**Details:** [[OLMo 3 Think - RLVR]]
**Related:** [[Verifier]], [[LM Judge]], [[GRPO]], [[Reward]]
