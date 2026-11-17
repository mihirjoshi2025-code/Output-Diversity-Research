---
tags: [olmo3, dpo, delta-learning, reading, week1]
source: "OLMo 3 report §4.3, §4.5, Table 19, Table 21, Table 22, Table 48, A.6.2"
---

# OLMo 3 Think: DPO (Dolci-Think-DPO, "Delta Learning")

> [!summary] The whole note in 4 lines
> - **DPO = learning from "answer A is better than answer B."**
> - "Better" = written by big **Qwen3-32B**; "worse" = written by tiny **Qwen3-0.6B**. Nobody actually judged them.
> - DPO pushes up **whatever the better answers have more of**. If that's Elara and lighthouses, it pushes those, even though only **8%** of the data is Core.
> - **Prime suspect.** Test: score the "better" and "worse" sides of the DPO data separately.

Part of [[OLMo 3 Think - Overview]]. Previous: [[OLMo 3 Think - SFT]]. Next: [[OLMo 3 Think - RLVR]]. Definition: [[Direct Preference Optimization (DPO)]].

**This is the stage most relevant to our Question 3.**

---

## The plain-English version

### What DPO is
**Direct Preference Optimization = learning from "A is better than B."** The model isn't shown one perfect answer to copy. It's shown **two** answers to the same prompt:
- a **chosen** one (labelled better)
- a **rejected** one (labelled worse)

and trained to make the chosen kind *more* likely and the rejected kind *less* likely. These are called [[Preference Pair|preference pairs]].

Think of a writing teacher who never writes anything. They just hold up two essays and say "this one's better," thousands of times. You learn to lean toward whatever the "better" essays have in common.

### How it works
1. Start from the SFT checkpoint. A frozen copy of it is kept as the **reference** ([[Policy|reference model]]).
2. For each pair, ask: compared with the reference, how much more likely is the model now to write the chosen answer than the rejected one?
3. Nudge the model so that gap grows.
4. A setting called [[Beta (DPO)|β]] stops it from drifting too far from the reference.

### Why that is different from SFT, and why it matters
SFT copies the *whole* answer, so it learns the general mix. DPO only learns from the **difference** between chosen and rejected. Any feature that appears **more often on the chosen side** gets pushed up, even if it's rare overall.

Example: suppose only 8% of DPO stories are Core stories, but nearly all of them are on the chosen side. To DPO, "Core-ness" looks like a sign of quality, so it pushes Core-ness up. That could take the model far beyond 8%. **This is how a method can *amplify* a pattern instead of copying it.**

### How Ai2 decided which answer was "better"
They didn't judge at all. They used a shortcut called [[Delta Learning]]:
- **Chosen** = answer from **Qwen3-32B**, a big, strong model.
- **Rejected** = answer from **Qwen3-0.6B**, a tiny, weak model.

The big model is *assumed* to be better. That's usually true for math and reasoning. But it also means DPO rewards **anything** Qwen3-32B does differently from Qwen3-0.6B, including style habits like favourite names or settings.

### How it connects to the paper
- Core share of the DPO training data = **8.0%** (Table 2).
- If the DPO checkpoint writes **far more** than 8% Core stories, and far more than the SFT checkpoint, DPO amplified the pattern. That is the setup for Question 3.
- **Direct test we can run:** score the chosen and rejected stories in `dolci-think-dpo-7b.jsonl` **separately**. If chosen stories are noticeably more Core than rejected ones, DPO was literally being told "Core = better."
- The report itself says the DPO model has **lower [[Entropy|entropy]]** (less varied choices) than SFT, which is consistent with narrowing ([[Mode Collapse]]).

---

## Technical details (optional: skip on a first read)

### How pairs were built (§4.3.1)
- **Chosen:** Qwen3-32B (thinking). **Rejected:** Qwen3-0.6B (thinking).
- **No judge and no human comparison.** The bigger model's response is labelled better by construction.
- **Delta Learning idea:** what matters is the quality *gap* between chosen and rejected, not how good the chosen response is on its own.
- Topic and identity filtering were applied to **chosen responses only**. Rejected responses were left unfiltered on purpose.
- Prompts were [[Decontamination|decontaminated]] against the evaluation sets.

### Why they did this (Table 21)
Further SFT on Qwen3-32B's answers *hurt* the model, but DPO against Qwen3-0.6B *helped*:

| Model | Avg benchmark score |
|---|---|
| Dev 7B SFT checkpoint | 70.3 |
| + continued SFT on chosen (just copy Qwen3-32B) | 64.5 |
| + Delta Learning DPO (prefer Qwen3-32B over Qwen3-0.6B) | **72.9** |

Plain English: copying had stopped helping, but *contrasting* good vs bad still taught the model something.

### Prompts (Table 19)
200K pool total. Mostly taken from Dolci Instruct SFT prompts, plus DaringAnteater and UltraFeedback.
Likely sources of story or creative prompts:
- **[[WildChat]]: 40,701**
- **UltraFeedback: 32,778** (not used in SFT)
- OpenAssistant: 1,762
- DaringAnteater: 1,089 (not used in SFT)

The prompt mix was chosen from the three best prompt distributions in the Instruct experiments, regenerated with the thinking Qwen models.

### Training settings (Table 48, 7B)
| Setting | Value | Plain English |
|---|---|---|
| Pairs | 150K | Number of A-vs-B comparisons used |
| [[Epoch\|Epochs]] | 1 | Each pair seen once |
| [[Beta (DPO)\|β]] | 5 | How tightly the model is tied to the SFT starting point |
| [[Learning Rate]] | 8.0 × 10⁻⁸, linear decay, 0.1 warmup | Tiny steps, ramped up then down |
| [[Batch Size]] | 128 | Pairs averaged per update |
| Max sequence length | 16K | Longest answer considered ([[Context Length]]) |
| [[Loss]] | Length-normalized DPO | Long answers aren't favoured just for length |

**[[Early Stopping|Early stopping]] matters:** performance rises up to a task-dependent point and then drops, so they swept dataset size and learning rate. DPO is a strong push; too much of it overshoots.

### Findings (§4.5)
- The DPO model has **lower [[Entropy|entropy]]** than SFT, but **higher [[Pass@k|pass@k]]** on AIME (Figure 20).
- RL starting from DPO beats RL starting from SFT (Table 22): SFT+DPO+RLVR averages 74.1 vs SFT+RLVR 71.9.

## Why it matters for us (my inference)
DPO pushes the model toward **whatever Qwen3-32B's responses share that Qwen3-0.6B's don't**. If Qwen3-32B's stories favour Elara, lighthouses and so on, DPO rewards those features directly, even though the Core rate in the DPO data is only 8.0%.

**Test:** score the chosen and rejected sides of `dolci-think-dpo-7b.jsonl` separately.
