---
tags: [vocab]
aliases: [pass@k, pass at k, pass@1]
---
# Pass@k

**In one line:** the chance that at least one of k attempts gets the answer right.

**Example:** pass@1 = right on the first try. pass@32 = right in at least one of 32 tries. High pass@k with low pass@1 means "it *can* do it, just not reliably."

**Why the report mentions it:** the DPO model has lower [[Entropy|entropy]] but *higher* pass@k on AIME math. In other words, less variety didn't hurt its ability to find right answers on math. For stories, where there is no "right answer," less variety simply means more repetition.

**Related:** [[Entropy]], [[Benchmark]]
