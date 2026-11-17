---
tags: [vocab]
aliases: [LM judge, LLM judge, LLM-as-a-judge, judge, AI judge]
---
# LM Judge (LLM-as-a-judge)

**In one line:** using one AI model to grade another AI model's answers.

**Plain explanation:** For tasks with no single right answer (stories, advice, chat), you can't check correctness with a program. Instead you give another model the prompt and the answer and ask it: "Rate this from 1 to 10." Its rating becomes the [[Reward|reward]].

**In OLMo 3:** the RL judge is **Qwen3-32B** (thinking turned off). It rates "helpfulness, relevance, accuracy, **creativity**, appropriate level of detail" and instruction following on a 1-10 scale.

**Why it matters for us:** the judge has its own taste. If Qwen3-32B thinks "Elara the lighthouse keeper" sounds creative and polished, RL rewards it, and the model being trained drifts toward it. Notice that Qwen3-32B is *also* the "chosen" model in DPO. The same model's taste may be pushed in twice.

Benchmarks use judges too: AlpacaEval uses GPT-4.1 as judge (see [[Benchmark]]).

**Related:** [[Reward]], [[RLVR]], [[Verifier]], [[Reward Hacking]]
