---
tags: [olmo3, reading, week1]
source: "OLMo 3 technical report, arXiv 2512.13961v2 (Ai2, Apr 2026), §4 and Appendix A.6-A.8"
---
# OLMo 3 Think: post-training overview

> [!summary] The whole note in 5 lines
> - OLMo is trained in steps: **read everything** (pretraining) → **copy examples** (SFT) → **pick the better of two answers** (DPO) → **practice with a grader** (RL).
> - Ai2 published the model **after each step**, so we can test each one.
> - Training data is only **3% → 8% → 11%** Core stories, but the final model writes **73%** Core stories. Something **amplified** them.
> - Main suspect: **DPO**. Second suspect: **RL**. Both lean on **Qwen3-32B's** taste.
> - Our job: fill in the Core % for each checkpoint and see where the jump happens.

Notes on the post-training of **Olmo-3-7B-Think**, written for the Core-stories project. Stage details:
- [[OLMo 3 Think - SFT]]
- [[OLMo 3 Think - DPO]]
- [[OLMo 3 Think - RLVR]]

New to the terms? Every linked word has a plain-English note in `Vocab/`. Start at [[_Vocab Index]].

---

## The plain-English version

### What is OLMo 3?
OLMo 3 is a [[Language Model|language model]] (an AI that writes text by predicting the next word) made by the Allen Institute for AI (Ai2). Unusually, Ai2 publishes **everything**: the model at each stage of training ([[Checkpoint|checkpoints]]) and the exact training data. That is why we can ask *when* in training something happens. With ChatGPT or Claude you only ever see the finished product.

### How does a model like this get made?
Think of it as raising a writer in four stages:

1. **[[Pretraining]]: reading the whole library.** The model reads trillions of words and learns to predict the next word. The result, the [[Base Model|base model]], knows a lot but doesn't follow instructions. It just continues text.
2. **[[Supervised Fine-Tuning (SFT)|SFT]]: copying good examples.** Show it millions of "question → ideal answer" examples (mostly written by *other* AIs) and train it to copy them. Now it acts like an assistant. *Lesson type: imitation.*
3. **[[Direct Preference Optimization (DPO)|DPO]]: learning "this is better than that."** Show it pairs of answers, one good and one worse, and push it toward the good one. *Lesson type: preference.*
4. **[[RLVR]]: practice with a grader.** The model writes its own answers, a grader scores them, and the model shifts toward high scores. *Lesson type: reward.*

Steps 2-4 together are called [[Post-training|post-training]]. We have a checkpoint after each of steps 2, 3 and 4.

### Why this matters for our question
The paper (*Elias in the Lighthouse, Again?*) found that LLMs keep writing the same kind of story: [[Core Stories|"Core" stories]] with characters like Elara or Elias, jobs like lighthouse keeper, and so on. In OLMo's case:

|                                         | SFT  | DPO  | RL    | Final model's stories |
| --------------------------------------- | ---- | ---- | ----- | --------------------- |
| Core share in the **training data**     | 3.0% | 8.0% | 10.9% | —                     |
| Core share in **what the model writes** | ?    | ?    | —     | **73.1%**             |

The training data has only a few Core stories, yet the finished model writes them almost three quarters of the time. Something in training **amplified** the pattern. Our job is to fill in the "?" cells by running the same prompts on each checkpoint and seeing where the jump happens.

**Why the method matters, in one sentence each:**
- **SFT** copies. A perfect copier would write about as many Core stories as it saw (~3%). A big jump here would be surprising.
- **DPO** learns from *differences* between good and bad answers. If "good" answers happen to be more Core-ish, DPO can push Core stories well beyond their 8% share. **Prime suspect.**
- **RL** rewards the model for its *own* answers that a judge likes. If the judge likes Core stories, the model writes more of them, then gets rewarded again, and so on. **Second suspect.**

### Who "taught" OLMo at each stage
OLMo's training answers were mostly written by other AIs ([[Teacher Model|teachers]]), so OLMo may have inherited their taste:

```
SFT:  copies DeepSeek R1's chat answers
DPO:  prefers Qwen3-32B's answers over Qwen3-0.6B's
RL:   graded by Qwen3-32B acting as a judge
```

Qwen3-32B shows up **twice**. If Qwen3-32B loves lighthouse stories, OLMo was pushed toward them in both DPO and RL.

---

## Technical details (optional: skip on a first read)

### Pipeline

```
Base ──► SFT ──► DPO ──► RLVR ──► Olmo-3-7B-Think
     Dolci-Think-SFT  Dolci-Think-DPO  Dolci-Think-RL
```

| Stage | Data ([[Dolci]]) | Where story-style text comes from | Our checkpoint |
| ----- | --------------------------------------- | ------------------------------------------------- | --------------------- |
| SFT | ~2.27M examples, mostly math and code | Chat answers written by **DeepSeek R1** | `Olmo-3-7B-Think-SFT` |
| DPO | 150K [[Preference Pair\|pairs]] (7B) | Chosen = **Qwen3-32B**, rejected = **Qwen3-0.6B** | `Olmo-3-7B-Think-DPO` |
| RLVR | ~105K prompts, ~21K general chat | [[Reward]] from a **Qwen3-32B [[LM Judge]]** | `Olmo-3-7B-Think` |

The report says each stage gives consistent gains, and that RL works better after DPO than straight after SFT (Table 22, Figure 19).

### Benchmark scores by stage (Table 15, 7B, mean of 3 runs)
[[Benchmark|Benchmarks]] are standard tests. Higher = better. None of them measure story variety.

| Benchmark | What it tests | SFT | DPO | Final |
|---|---|---|---|---|
| AlpacaEval 2 LC | General chat quality (AI-judged) | 43.9 | **50.6** | 52.1 |
| IFEval | Following exact rules | 77.9 | 75.9 | **88.2** |
| IFBench | Harder rules | 30.0 | 28.3 | 41.6 |
| AIME 2025 | Hard math | 57.6 | 62.7 | 64.6 |
| MATH | Math | 94.4 | 92.4 | 95.1 |
| Safety | Refusing harmful requests | 65.8 | 67.7 | 70.7 |

- The biggest chat-quality jump is at **DPO**. RL mostly improves [[Instruction Following|instruction following]].
- §4.5: the DPO model shows **lower [[Entropy|entropy]]** than SFT (its choices are less varied), yet higher [[Pass@k|pass@k]] on AIME. This is the report's only direct comment on output diversity, and it hints that DPO narrows outputs ([[Mode Collapse]]).

### Evaluation settings the report uses (Table 16, A.8.1)
- [[Temperature]] **0.6**, [[Top-p|top_p]] **0.95**, max **32,768** [[Token|tokens]], zero-shot (no examples in the prompt).
- [[Reasoning Trace|Thinking traces]] (`<think>…</think>`) are stripped before scoring, as in our rule to score only the story.
- AlpacaEval 2 LC uses GPT-4.1 as judge. The appendix openly admits AlpacaEval rewards "flair" and over-optimized style ([[Reward Hacking]]).

## Implications for our project (my inferences, not claims in the report)
- **Teacher shift across stages:** DeepSeek R1 (SFT) → Qwen3-32B preferred over Qwen3-0.6B (DPO) → Qwen3-32B as judge (RL). Qwen3-32B's story habits could be reinforced twice.
- **Prediction:** the Core rate should jump at DPO and may rise again at RL.
- **Token cap:** the report evaluates at 32K tokens, and RL [[Rollout|rollouts]] averaged over 10K. Our 4,096 cap may truncate some outputs, especially from SFT ([[Context Length]]).
- **Sampling:** the old OLMo run's settings are unknown. Whatever we choose, keep it identical across checkpoints.

## Questions for the mentor
- Score stories from **Qwen3-32B vs Qwen3-0.6B** directly? They are the DPO chosen and rejected sources.
- Score the **chosen vs rejected** sides of `dolci-think-dpo-7b.jsonl` separately. A higher Core rate on the chosen side would support DPO amplification.
- Keep the 4,096-token cap, or move closer to the report's settings?
