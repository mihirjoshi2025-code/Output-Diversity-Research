---
tags: [vocab]
aliases: [reasoning trace, think block, thinking trace, "<think>", chain of thought, reasoning, CoT]
---
# Reasoning Trace (`<think>` block)

**In one line:** the "scratch work" a thinking model writes before its real answer, wrapped in `<think> … </think>`.

**Example output:**
```
<think>The user wants a story. Maybe a lighthouse keeper named Elias who... </think>
Elias had kept the lighthouse for forty years...
```
Everything inside `<think>` is planning. Everything after `</think>` is the answer.

**Why it matters for us:**
- **Rule: score only the story.** The reasoning might *mention* "Elara" while brainstorming and then not use her. Counting that would inflate Core rates, so we strip everything up to `</think>` before scoring.
- The reasoning still uses up [[Token|tokens]]. With a 4,096 cap, long thinking can cut off the story (`finish_reason: length`).
- We save the raw reasoning anyway: it may show *when* the model decides on a lighthouse.

**Related:** [[Thinking Model]], [[Context Length]]
