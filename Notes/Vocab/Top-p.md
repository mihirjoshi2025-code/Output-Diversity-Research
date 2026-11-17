---
tags: [vocab, sampling]
aliases: [top-p, top_p, nucleus sampling, top-k, min-p]
---
# Top-p (Nucleus Sampling)

**In one line:** only consider the most likely next tokens whose probabilities add up to p, and ignore the long tail of unlikely ones.

**Example:** with top_p = 0.95, the model sorts the candidates by probability and keeps adding them until they cover 95%. It samples only from that set, which blocks rare junk tokens.

**Cousins:** **top-k** (only the k most likely tokens) and **min-p** (drop tokens below a fraction of the top token's probability). llama-server has defaults for all of these, so we must record exactly what was used.

**OLMo report setting:** [[Temperature|temperature]] 0.6, top_p 0.95.

**Related:** [[Temperature]]
