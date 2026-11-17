---
tags: [vocab]
aliases: [context length, context window, sequence length, max tokens, token cap, max_tokens]
---
# Context Length (and Token Cap)

**In one line:** the maximum number of [[Token|tokens]] a model can handle at once (input + output).

- **Context window / sequence length:** how much the model can "see." OLMo trained with 16K-32K.
- **Max output tokens (token cap):** how much we let it *write*. The old run: **4,096**. The report's evaluations: **32,768**.

**finish_reason:** when generation ends, the server says why:
- `stop`: the model finished naturally.
- `length`: it hit the cap and was cut off. The story may be missing or unfinished.

**Related:** [[Reasoning Trace]], [[Rollout]]
