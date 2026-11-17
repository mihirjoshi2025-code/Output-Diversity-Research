---
tags: [vocab, project]
aliases: [PPM, parts per million]
---
# PPM (Parts Per Million)

**In one line:** how many times a word appears per 1,000,000 words of text.

**Example:** "Elara" at 6,625.5 PPM in OLMo's stories means about 6.6 mentions per 1,000 words of story text, *on average across all stories*. It's uneven: a story starring Elara mentions her many times, and many stories don't mention her at all.

PPM lets you compare corpora of different sizes. Our scorer computes it as `hits × 1,000,000 / total words`.

**Related:** [[Core Stories]]
