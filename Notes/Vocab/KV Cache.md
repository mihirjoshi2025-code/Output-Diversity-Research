---
tags: [vocab, inference]
aliases: [KV cache, kv cache, context memory]
---
# KV Cache

**In one line:** each story's private memory of everything it has written so far.

**Plain explanation:** To pick word 501, the model needs information about words 1–500. Instead of re-reading them every step, it saves a summary of each word (its "keys" and "values") in the **KV cache**. It costs memory: about **0.5 MB per token** for OLMo 7B, so `-c 16384` reserves ~8 GB.

**Why it matters for us:** with `--parallel 4`, each of the 4 stories gets its own slice of the cache, which is why they never mix ([[Parallel Generation - 4 Stories at Once]]). A bigger `-c` uses more memory.

**Related:** [[Context Length]], [[Token]]
