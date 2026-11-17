---
tags: [vocab]
aliases: [WildChat, OpenAssistant, UltraFeedback]
---
# WildChat (and other prompt sources)

**In one line:** a big collection of *real* conversations people had with ChatGPT, released for research.

**Why it matters:** WildChat is where most "write me a story" prompts in OLMo's training data come from, since real users ask for stories a lot. Ai2 noticed some character names were overused in WildChat role-play prompts (Natsuki, Monika...) and capped each at 10 for RL. So they *were* aware that name concentration can be a problem, at least in prompts.

**Related sources:**
- **OpenAssistant:** volunteer-written conversations.
- **UltraFeedback:** a preference dataset; prompts reused for DPO.

**Related:** [[Dolci]], [[Synthetic Data]]
