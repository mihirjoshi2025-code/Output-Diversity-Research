---
tags: [vocab]
aliases: [delta learning, Delta Learning]
---
# Delta Learning

**In one line:** the idea that preference training works because of the *gap* between the chosen and rejected answers, not because the chosen answer is excellent.

**Plain explanation:** "Delta" means difference. OLMo 3 built its DPO pairs cheaply: chosen = answer from a big model (Qwen3-32B), rejected = answer from a tiny model (Qwen3-0.6B). No human or judge compared them. The big model is just *assumed* to be better. Ai2 found this helps even when copying the big model's answers with SFT doesn't (it actually hurt in their test: 70.3 → 64.5, while DPO went 70.3 → 72.9).

**Analogy:** you learn what makes a good essay faster by comparing a good essay with a bad one than by reading only good essays.

**Why it matters for us:** DPO then pushes OLMo toward *anything* Qwen3-32B does more than Qwen3-0.6B. That includes being smarter, but also any **stylistic habits** of Qwen3-32B, such as favourite names or settings. If Qwen3-32B writes more lighthouse stories than Qwen3-0.6B, lighthouse stories get rewarded, whether or not they are actually better.

**Related:** [[Direct Preference Optimization (DPO)]], [[Preference Pair]], [[Teacher Model]]
