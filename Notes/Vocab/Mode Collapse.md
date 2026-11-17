---
tags: [vocab]
aliases: [mode collapse, low diversity, output diversity, diversity, homogenization]
---
# Mode Collapse (Low Diversity)

**In one line:** when a model keeps producing the same few kinds of output instead of a wide variety.

**Plain explanation:** Ask a person for "a story" 100 times and you'd get 100 different stories. Ask a heavily post-trained model and you may get "Elara the lighthouse keeper" over and over. The model's options have "collapsed" onto a favourite (a *mode*, the statistics word for the most common value).

**This is the paper's topic.** *Elias in the Lighthouse, Again?* measures exactly this collapse in LLM stories. Our project asks **which training stage causes it** in OLMo 3.

**Known suspects:** preference training ([[Direct Preference Optimization (DPO)|DPO]], [[RLHF]]) and [[Reinforcement Learning|RL]] have both been linked to lower diversity (Kirk et al. 2024). The OLMo report notes the DPO model has lower [[Entropy|entropy]] than SFT.

**Related:** [[Core Stories]], [[Entropy]], [[Reward Hacking]]
