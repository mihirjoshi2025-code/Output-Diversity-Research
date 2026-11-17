---
tags: [vocab, project]
aliases: [Core, Core story, Core stories, Core words]
---
# Core Stories

**In one line:** stories containing at least one of 11 words the paper found LLMs overuse.

| Group | Words |
|---|---|
| Names | Elias, Mara, Elara |
| Professions | keeper, baker, mayor, clockmaker, fisherman, librarian, conductor |
| Setting | lighthouse |

**How we measure it:** `helper_scripts/core_rates.py` does whole-word, case-insensitive matching on **the story only** (not the prompt, not the [[Reasoning Trace|reasoning]]).
- **% with Core:** share of stories containing ≥1 Core word. OLMo final model: **73.1%**.
- **[[PPM]]:** how often each word appears per million words.

**Known scorer limits:** "goalkeeper" won't match (good), but "Mayor" as a surname or "conductor" in a physics sense would (false positives).

**The project's question:** at which training stage ([[Supervised Fine-Tuning (SFT)|SFT]], [[Direct Preference Optimization (DPO)|DPO]], [[RLVR]]) does the Core rate jump?

**Related:** [[Mode Collapse]], [[PPM]]
