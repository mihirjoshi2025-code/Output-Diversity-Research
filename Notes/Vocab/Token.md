---
tags: [vocab]
aliases: [tokens, token, tokenizer]
---
# Token

**In one line:** the small chunk of text a [[Language Model|language model]] reads and writes, usually a word or part of a word.

**Plain explanation:** Models don't see letters or whole words. They see tokens. Common words are often one token ("the"), and rare words get split ("lighthouse" might be "light" + "house"). A rough rule for English: 1 token ≈ ¾ of a word, so 100 tokens ≈ 75 words.

**Why it matters:** limits are counted in tokens, not words. Examples:
- The old OLMo run had a **4,096-token cap**, about 3,000 words, and that includes the hidden [[Reasoning Trace|reasoning]]. If the model "thinks" for 3,500 tokens, only about 600 are left for the story.
- Speed is measured in **tokens per second**.

**Related:** [[Context Length]], [[Reasoning Trace]]
