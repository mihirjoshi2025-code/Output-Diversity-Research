---
tags: [vocab]
aliases: [decontamination, decontaminated, contamination, n-gram]
---
# Decontamination

**In one line:** removing training examples that overlap with test questions, so the model can't just memorize benchmark answers.

**How OLMo does it:** compares 8-word sequences ("8-grams"); if 50% or more of an example's 8-grams appear in a test set, it's removed.

**For us:** not directly relevant to stories, but it explains a filtering step you'll see in the report.

**Related:** [[Benchmark]]
