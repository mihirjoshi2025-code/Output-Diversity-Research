---
tags: [vocab]
aliases: [thinking model, Think model, Think, reasoning model, Instruct]
---
# Thinking Model (Think vs Instruct)

**In one line:** a model trained to "think out loud" in a [[Reasoning Trace|reasoning trace]] before answering.

**Plain explanation:** OLMo 3 has two versions built from the same [[Base Model|base model]]:
- **Think:** writes long `<think>…</think>` reasoning, then answers. Better at math and code.
- **Instruct:** answers directly. Faster, aimed at chat.

**In our project:** we study **Think** only (`Olmo-3-7B-Think-*`). Ignore the `dolci-instruct-*` files for now.

**Related:** [[Reasoning Trace]], [[Checkpoint]]
