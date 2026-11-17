---
tags: [olmo3, sft, reading, week1]
source: "OLMo 3 report §4.2, Table 17, Table 18, Table 47, A.7.1"
---

# OLMo 3 Think: SFT (Dolci-Think-SFT)

> [!summary] The whole note in 4 lines
> - **SFT = learning by copying** example answers, word for word.
> - The story-style examples were written by **DeepSeek R1** (another AI).
> - Only **3%** of the SFT data is Core stories, and copying shouldn't exaggerate that.
> - So SFT is our **baseline**: if its Core % is already high, DPO/RL aren't the whole story.

Part of [[OLMo 3 Think - Overview]]. Next stage: [[OLMo 3 Think - DPO]]. Definition: [[Supervised Fine-Tuning (SFT)]].

---

## The plain-English version

### What SFT is
**Supervised fine-tuning = learning by copying.** You give the model a huge stack of examples, each one a prompt plus a model answer, and train it to reproduce the answers word by word.

Like a student who studies worked solutions until they can produce that kind of answer on their own.

### How it works
1. Take an example: prompt "Write a story about the sea" plus a finished answer (with a `<think>` planning section, then the story).
2. Have the model predict the answer one [[Token|token]] at a time.
3. Whenever it guesses wrong, measure how wrong ([[Loss|loss]]) and nudge its [[Parameters|internal numbers]] so the right token is a bit more likely next time.
4. Do this for ~2.27 million examples, twice through ([[Epoch|2 epochs]]).

Before SFT, OLMo was a [[Base Model|base model]] that just continued text. After SFT it behaves like an assistant that thinks in `<think>` tags and then answers. This is the first checkpoint we test.

### Who wrote the examples?
Mostly **other AIs** ([[Teacher Model|teacher models]], [[Synthetic Data|synthetic data]]). Real people wrote many of the *prompts* (e.g. from [[WildChat]]), but the *answers* came from:
- **DeepSeek R1** for chat (which includes story requests).
- **QwQ-32B** for math, code and science.

So SFT-stage OLMo writes stories the way **DeepSeek R1** does.

### How it connects to the paper
- Only **3.0%** of the SFT training stories are Core (paper Table 2).
- SFT is imitation, and imitation tends to reproduce the mix it saw. It has no built-in reason to *exaggerate* a pattern.
- **Expectation:** the SFT checkpoint's Core rate should be fairly low. If it's already high (say 50%+), the pattern doesn't come from DPO/RL alone. It might come from the [[Base Model|base model]], or SFT might generalize differently than "copy the mix."
- Either way, SFT is the **baseline** that DPO and RL are measured against.

---

## Technical details (optional: skip on a first read)

### Data (Table 17, 7B counts)
About **2.27M** examples in total.

| Category | Sources | Count | Who wrote the responses |
|---|---|---|---|
| Math | OpenThoughts3+ math, SYNTHETIC-2 | ~857K | QwQ-32B (OpenThoughts), SYNTHETIC-2 verified |
| Code | Python Algorithms, OpenThoughts3+ code, Nemotron | ~669K | QwQ-32B (filtered by tests), Nemotron |
| Precise IF | Persona IF, Dolci Precise IF | ~359K | QwQ-32B, kept only if constraints verified |
| **Chat** | **WildChat (83,054), OpenAssistant (6,800)** | ~90K | **DeepSeek R1** |
| Safety | CoCoNot, WildGuardMix, WildJailbreak | ~90K | DeepSeek R1 |
| Science | OpenThoughts3+ STEM | ~99K | QwQ-32B |
| Other | Aya (multilingual), TableGPT, identity prompts | ~104K | DeepSeek R1 |

"Precise IF" = [[Instruction Following|precise instruction following]] (obeying exact rules like "use 3 bullets").

**For us:** story requests mostly come in through WildChat and OpenAssistant, so SFT-stage story style is **DeepSeek R1's**. Chat is only ~4% of the SFT data. Most of SFT is math and code.

### Filtering (§4.2.1, A.7.1)
What Ai2 threw out before training:
- Non-commercial licenses and incomplete reasoning (`<think>` without `</think>`).
- Mentions of other model developers and date cutoffs (so OLMo doesn't claim to be ChatGPT).
- **Repetition filter:** sentences or paragraphs repeated 10+ times, phrases 50+ times. (This catches *within-answer* repetition, not the same story across many answers.)
- Responses with ≥5% Chinese characters.
- **Topic filtering** with the OpenAI query taxonomy: irrelevant WildChat topics (e.g. image generation, basic greetings) were cut or downsampled.
- [[Decontamination|Decontaminated]] against evaluations with 8-gram matching at a 0.5 overlap threshold.

### Mixing ablation (Table 18)
An [[Ablation|ablation]]: each category was added to a 100K "base mix" to see its effect. Adding **WildChat + OpenAssistant** raised AlpacaEval from 19.0 to 38.5. Chat data is what produces the chat style.

### Training (§4.2.2, A.6.1)
- OLMo-core, **2 epochs**, 32K sequence length ([[Context Length]]), 1M-token [[Batch Size|batches]] (7B), 45.4B total tokens.
- **The final SFT checkpoint is a [[Model Merging|merge ("soup")]] of two runs with different [[Learning Rate|learning rates]]**, made with mergekit.
- Checkpoints were also chosen by qualitative "vibe tests" (people trying the model and judging how it feels).

## Things to check in our runs
- Does SFT reliably close its `<think>` block? Training data with unclosed blocks was filtered out, so it should.
- Does SFT write a story at all under our system prompt?
- Does it hit the 4,096-token cap (`finish_reason: length`) more often than the later checkpoints?
