---
tags: [vocab]
aliases: [LLM, LLMs, language model, language models, large language model]
---
# Language Model (LLM)

**In one line:** a program that reads text and guesses the next piece of text, one piece at a time.

**Plain explanation:** A language model is trained on a huge amount of text, and its only skill is predicting what comes next. Give it "The cat sat on the" and it gives a probability to every possible next word ("mat" high, "refrigerator" low). To write a whole story, it picks a next word, adds it to the text, and repeats thousands of times. Everything else, such as chatting, answering questions or writing stories, is built on top of this one skill.

**Analogy:** your phone's autocomplete, made enormously bigger and trained on much more text.

**In our project:** OLMo 3 7B Think is the language model we study. "7B" means it has about 7 billion [[Parameters|parameters]].

**Related:** [[Token]], [[Parameters]], [[Base Model]], [[Pretraining]]
