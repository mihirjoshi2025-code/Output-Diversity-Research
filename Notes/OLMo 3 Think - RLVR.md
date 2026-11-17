---
tags: [olmo3, rlvr, grpo, reading, week1]
source: "OLMo 3 report §4.4, §4.5, Figure 16, Figure 40, Table 20, Table 49"
---

# OLMo 3 Think: RLVR (Dolci-Think-RL, OlmoRL)

> [!summary] The whole note in 4 lines
> - **RL = trial and error.** The model writes 8 answers, a grader scores them, and it leans toward the above-average ones.
> - Math and code get checked by programs; **stories get graded by Qwen3-32B** (the same model DPO favoured).
> - It trains on its **own** answers, so habits can snowball, and nothing holds it back (no KL penalty).
> - **Second suspect.** Compare the DPO checkpoint with the final model to see how much RL added.

Part of [[OLMo 3 Think - Overview]]. Previous: [[OLMo 3 Think - DPO]]. Definitions: [[RLVR]], [[Reinforcement Learning]], [[GRPO]].

---

## The plain-English version

### What RL is
**Reinforcement learning = learning by trial and error.** Nobody shows the model a correct answer. Instead:
1. The model writes its **own** answers to a prompt ([[Rollout|rollouts]]).
2. Something scores each answer ([[Reward|reward]]).
3. The model shifts toward the answers that scored well.

Like training a dog with treats: you don't demonstrate the trick, you reward it when it gets it right.

### What "VR" (verifiable rewards) means
For many tasks a computer program can *check* the answer ([[Verifier|verifiers]]):
- **Math:** is the final number correct?
- **Code:** do the tests pass?
- **Rules:** did it really use exactly 3 paragraphs?

Those give a clean right/wrong score with no opinions involved.

**But stories can't be checked by a program.** For chat and creative prompts, OLMo uses an [[LM Judge|AI judge]] instead: **Qwen3-32B** reads the answer and rates it 1-10 on helpfulness, relevance, accuracy, **creativity**, and detail. So for our purposes, story prompts get an **opinion**, not a verifiable fact.

### How OLMo does it ([[GRPO]], "grading on a curve")
For each prompt the model writes **8** answers. Each gets a score. Answers **above the group average** are made more likely; answers **below** are made less likely ([[Advantage|advantage]]). Repeat for 1,400 training steps.

### Why RL could amplify Core stories
- **It feeds on its own habits.** The model is trained on stories *it* wrote. If after DPO it already likes lighthouse stories, it writes lots of them, and if the judge rates them slightly above average, they get reinforced again. Small preferences can snowball.
- **The judge is Qwen3-32B**, the same model whose answers were "chosen" in DPO. Its taste gets pushed in a second time.
- **No leash.** Many RL setups add a [[KL Penalty|KL penalty]] that stops the model drifting too far from where it started. **OLMo 3's RL has none**, so the model can move as far as the reward pulls it ([[Reward Hacking]]).

### How it connects to the paper
- Core share of the RL training data = **10.9%** (Table 2). Note that an RL dataset holds prompts (plus some reference answers), not the stories the model actually trained on, because the model writes its own answers during training. Check the `training_stories/` README to see exactly which text Table 2 scored for RL.
- The final model (after RL) writes **73.1%** Core stories.
- Comparing the DPO checkpoint with the final checkpoint tells us how much of the jump happens here. If DPO is already near 73%, RL just maintained it; if DPO is much lower, RL did a lot of the amplifying.

---

## Technical details (optional: skip on a first read)

### Data (Table 20, Think RL column)
About **105K prompts** in total.

| Domain | Sources | Count |
|---|---|---|
| Precise [[Instruction Following\|IF]] | IF-RLVR (up to 5 constraints from IFEval / IFBench-Train) | 30,186 |
| Math | Open-Reasoner-Zero, DAPO-Math, AceReason, KlearReasoner, OMEGA | ~30K |
| Code | AceCoder, KlearReasoner-Code, Nemotron, SYNTHETIC-2 | ~23K |
| **General chat** | **Tülu 3 SFT 7,129, WildChat-4.8M 7,129, Multi-Subject RLVR 7,129** | ~21K |

Only ~20% of RL prompts are general chat (where stories live), but those prompts get the judge reward.

#### General chat details (§4.4.2)
- **[[WildChat]]:** English only, prompts that don't need reasoning. Role-play is common, so **any single character was capped at 10 appearances**. Before the cap: Natsuki 1,284, Monika 1,243, Sayori 1,076, Yuri 957, Sakura 453, MC 424 (footnote 38). Ai2 noticed name concentration in the *prompts*. (Interesting for us: they knew repeated names were a risk, but only checked prompts, not outputs.)
- **Tülu 3:** rewritten by GPT-4.1 to extract reference answers. Kept only prompts where a Qwen 2.5 7B model's F1 against the reference averaged 0.1-0.8 (not too easy, not impossible).
- **Offline difficulty filter:** 8 rollouts per prompt from the DPO checkpoint at [[Temperature|temp]] 1.0. Prompts solved more than 62.5% of the time were removed (too easy to teach anything).

### Rewards (Figure 16)
| Domain | Who scores it | Reward |
|---|---|---|
| Math | [[Verifier]]: SymPy comparison with the reference | 0 / 1 |
| Code | Verifier: test cases | Pass rate or all-pass 0 / 1 |
| Instruction following | Verifier: constraint-checking functions | 1 only if all constraints met |
| **Chat with reference** | **[[LM Judge]] compares with the reference answer** | **[0, 1]** |
| **Chat, open-ended (stories land here)** | **LM judge, no reference** | **[0, 1]** |

**LM judge:** Qwen3-32B with thinking off, served by vLLM. Max input 32,768 tokens, response 2,048.
Judge prompt (Figure 40): rate "helpfulness, relevance, accuracy, **creativity**, appropriate level of detail" and instruction following, output a 1-10 score as JSON. The text reports rewards in [0, 1], so the score is presumably rescaled.

### Algorithm: OlmoRL ([[GRPO]] plus recent improvements)
- Groups where all rewards are equal are dropped (no [[Gradient|gradient]], nothing to learn). "Active sampling" refills the batch.
- Token-level [[Loss|loss]], so there is no per-sample length bias.
- **No [[KL Penalty|KL penalty]].** Nothing explicitly keeps the [[Policy|policy]] close to the DPO model.
- Clip-higher (0.2 lower, 0.272 upper) and truncated importance sampling (stability tricks).
- **No standard-deviation normalization** of [[Advantage|advantages]].

### Settings (Table 49, 7B Think)
| Setting | Value | Plain English |
|---|---|---|
| Training steps | 1,400 | Number of update rounds |
| [[Learning Rate]] | 1.0 × 10⁻⁶, constant | Step size, never decays |
| Prompts per [[Batch Size\|batch]] | 64 | Prompts per update |
| Group size | 8 | Answers written per prompt |
| Response length | 32,768 | Max tokens per answer |
| Sampling [[Temperature\|temperature]] | 1.0 | Model's natural randomness |

The 7B run took about 15 days on the old infrastructure (about 6 days on the new). Rollouts averaged over 10K [[Token|tokens]].

### Findings (§4.5)
- Rewards rise steadily in every domain. Instruction-following rises most steadily, code most slowly.
- **Training on one domain over-optimizes it ([[Reward Hacking]]):** RL on IFEval alone traded IFEval gains for lower AlpacaEval. Mixing domains with the LM-judge reward kept chat quality up.
- Mixed-data runs get lower training reward per domain but generalize better, which suggests less reward hacking.
- RL gives the biggest gain in precise instruction following (IFEval 75.9 → 88.2 for 7B).

## Why it matters for us (my inference)
- Story prompts in RL get **only** the Qwen3-32B judge's score. If that judge prefers certain story styles, RL reinforces the same Qwen3 preferences that DPO introduced.
- With no KL penalty, the model is free to drift as far as the reward pushes it.
- Instruction-following rewards (e.g. "use exactly 3 paragraphs") could also shape story *form*, though not the names or settings.
