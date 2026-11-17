---
title: "Which Training Stage Writes Like the Final Model?"
subtitle: "A worksheet on checkpoint prediction for the OLMo 3 Core-stories project"
author: "Prepared for Mihir Joshi · Cornell AI Research (mentors: Sil Hamilton and David Mimno)"
date: "October 29, 2026"
documentclass: article
classoption: [11pt]
geometry: [letterpaper, margin=1in]
mainfont: Charter
sansfont: Helvetica Neue
monofont: Menlo
monofontoptions: [Scale=0.88]
linkcolor: accent
urlcolor: accent
toccolor: black
colorlinks: true
toc: true
toc-depth: 2
numbersections: true
---

\newpage

# How to use this worksheet {-}

Your mentor suggested studying how similar the SFT and DPO stories are to the final (RL) model's
stories, and treating it **as a prediction task**. This worksheet teaches you everything you need
to build that analysis yourself in this repository: the statistics behind it, how to set up the
data so the answer means something, how to code it in scikit-learn, and how to read the results.

It's built around one idea:

> *If a classifier can't tell which checkpoint wrote a story, the two checkpoints write the same
> way. How well it can tell them apart measures how different they are.*

**How it is organized.** Sections 1 to 3 are the reasoning and the data plan. Sections 4 to 6 build
the pieces (features, model, splits). Section 7 is a hands-on warm-up on data you **already
have** (the 1,000 Instruct stories). Sections 8 and 9 cover the statistics and interpretation.
Section 10 is an optional second method. Section 11 is the plan for the real Think experiment.
Answers are in Appendix A.

**The boxes.**

::: {.concept}
The main ideas. If you remember nothing else, remember these.
:::

::: {.exercise title="0.1"}
Something to derive, compute or code. Try it before looking at Appendix A.
:::

::: {.checkpoint}
What a correct result looks like, so you know whether to move on.
:::

::: {.pitfall}
A mistake that would quietly make your result wrong.
:::

::: {.repo}
Where something lives in the repository, or which existing code to reuse.
:::

**What you need.** Python, pandas, and comfort reading the existing notebook in `notebooks/`
(`01_instruct_sft_vs_dpo_core_rates.ipynb`, called "notebook 01" below). New libraries: `scikit-learn` (the model),
`scipy` (tests), `matplotlib` (plots), all launched the same way as the existing notebook:

```bash
cd ~/"AI Research - Cornell"
uv run --no-project --with jupyterlab,pandas,scikit-learn,scipy,matplotlib jupyter lab
```

\newpage

# Where this fits in the project

## The research question, restated

The project asks: *at which post-training stage (SFT, DPO, RLVR) do OLMo 3 7B Think's Core stories
(Elias / Mara / Elara, keepers and bakers, lighthouses) appear or become much more common?* So far
you've measured that with **one number per story**: does it contain any of the 11 Core words?

That number is useful but narrow. Your Instruct results showed why:

- DPO's Core rate rose 44.4% $\to$ 53.4%, but almost all of that was *lighthouse* and *keeper*.
- Meanwhile **Clara** (not a Core word) became the top name, and **"Marrow's End"** went from 3
  stories to 56. The Core score can't see either.

The prediction task measures **everything that differs** between two checkpoints' stories, not just
11 words. Then you can ask how much of that difference *is* the Core words.

## The questions it can answer

1. **Distance.** How different are SFT stories from RL stories? DPO from RL? If DPO $\approx$ RL but
   SFT $\neq$ RL, the final model's style was set during DPO.
2. **Content.** *What* separates them? The classifier's most important words tell you.
3. **Role of the Core words.** Hide the Core words and retrain. If accuracy barely drops, the stages
   differ in ways the Core list doesn't capture.
4. **Direction.** Where do DPO stories sit on the line from SFT to RL? Partway, all the way, or off
   to one side?
5. **Shape of the change** (Section 10). Did RL move the model somewhere new, or *narrow* it onto
   DPO's favourite patterns (mode collapse)?

## Where the code will live

::: {.repo title="project map"}
```text
AI Research - Cornell/
|-- helper_scripts/
|   |-- core_rates.py        REUSE: WORDS, PATTERN, CORE, score_core()
|   |-- download_models.py   REUSE: MODELS = stage -> (repo, revision)
|   |-- generate_stories.py  makes the JSONL files you'll classify
|   `-- stage_data.py        NEW (you write): load + frozen split
|-- new_stories/data/        (gitignored, via data/)
|   |-- instruct-sft.jsonl   warm-up data, 500 stories
|   |-- instruct-dpo.jsonl   warm-up data, 500 stories
|   `-- think-*.jsonl        TO GENERATE: sft, dpo, rlvr (Sec. 11)
|-- notebooks/
|   |-- 01_instruct_sft_vs_dpo_core_rates.ipynb
|   |                        REUSE: wilson(), setup-check cell
|   `-- 02_stage_prediction.ipynb    NEW (you write)
|-- results/stage_prediction/
|                            NEW: test_keys.json, metrics.json
`-- worksheets/stage-prediction/     this worksheet
```
:::

The split of responsibilities is the same as in notebook 01: **reusable logic** (loading, splitting)
goes in `helper_scripts/` so it can be imported and tested; **analysis and results** go in the
notebook. The test split gets saved to `results/` so it is *frozen*: every later experiment uses
exactly the same held-out stories.

::: {.pitfall title="Project rules that apply here"}
From `CLAUDE.md`: **score only the story** (never the prompt or the `<think>` reasoning), and
**same setup for every checkpoint**. For a classifier the second rule is critical. Section 3 shows
that a classifier detects *any* difference, including a setup difference.
:::

\newpage

# The math: classifier two-sample tests

This section explains *why* classification accuracy measures similarity. It's the most important
theory in the worksheet.

## Stories as samples from distributions

Treat each checkpoint as a **probability distribution over stories**. When you run checkpoint
$A$ (say DPO) on our prompts with our sampling settings, each story is a random draw $x \sim P$.
The final RL checkpoint gives draws $x \sim Q$. The question "do DPO and RL write the same way?" is
the statistics question

$$
H_0:\ P = Q \qquad \text{vs.} \qquad H_1:\ P \neq Q,
$$

plus the follow-up *how far apart are they?* This is a **two-sample problem**. The difficulty is
that a story is a very high-dimensional object, so you can't just compare two histograms.

## The classifier trick

Pool the two samples and label them: $y = 0$ for stories from $P$, $y = 1$ for stories from $Q$,
with equal numbers of each. Train a classifier to predict $y$ from the story. Then:

- If $P = Q$, the story carries **no information** about its label. No classifier can do better
  than 50% on new data.
- If $P \neq Q$, some classifier can beat 50%, and the further apart they are, the better the best
  one can do.

This is the **classifier two-sample test** (C2ST; Lopez-Paz and Oquab, 2017). The next two results
make the second bullet precise.

## The best possible classifier

Write $p(x)$ and $q(x)$ for the probabilities of story $x$ under the two checkpoints. With equal
class sizes, the prior is $\Pr(y=1) = \tfrac12$, and Bayes' rule gives

$$
\Pr(y = 1 \mid x) \;=\; \frac{\tfrac12 q(x)}{\tfrac12 p(x) + \tfrac12 q(x)} \;=\; \frac{q(x)}{p(x) + q(x)}.
$$

The classifier that minimizes the error rate (the **Bayes classifier**) predicts $y=1$ exactly when
$q(x) > p(x)$. Its accuracy is

$$
\text{acc}^{*} \;=\; \tfrac12 \sum_x \max\{p(x), q(x)\}.
$$

Now define the **total variation distance**

$$
\TV(P, Q) \;=\; \tfrac12 \sum_x \lvert p(x) - q(x) \rvert \;\in\; [0, 1],
$$

which is 0 when the distributions are identical and 1 when they never produce the same story.

::: {.exercise title="2.1 Bayes accuracy equals a distance"}
Using the identity $\max\{a,b\} = \frac{a+b}{2} + \frac{\lvert a-b\rvert}{2}$, show that

$$
\text{acc}^{*} = \tfrac12 + \tfrac12\,\TV(P,Q).
$$

Then explain in one sentence why this means **any** classifier's true accuracy gives a lower bound
$\TV(P,Q) \ge 2\,\text{acc} - 1$.
:::

::: {.concept title="What the accuracy number means"}
A classifier's held-out accuracy is (an estimate of) a **lower bound on how different the two
checkpoints are**. 50% means "no difference *this classifier can find*", which isn't proof
of no difference. 100% means the checkpoints essentially never write stories that look alike. A
better classifier gives a tighter bound, but even a simple one gives a valid bound.
:::

::: {.exercise title="2.2"}
In the Instruct warm-up you'll see accuracies around 0.84. What lower bound on $\TV$ does that give?
What would an accuracy of 0.55 imply?
:::

## How much accuracy is "real"?

Accuracy is measured on a finite test set of $n$ stories, so it is noisy. Under $H_0$ every
prediction is a fair coin flip, so the number correct is $K \sim \mathrm{Binomial}(n, \tfrac12)$
with standard error

$$
\mathrm{SE}(\widehat{\text{acc}}) = \sqrt{\frac{\text{acc}(1-\text{acc})}{n}} \;=\; \sqrt{\frac{0.25}{n}} \quad \text{at acc} = 0.5.
$$

::: {.exercise title="2.3 Power with our sample sizes"}
The plan (Section 6) holds out 100 stories per checkpoint, so a two-way test set has $n = 200$.

(a) Compute the standard error at 0.5.

(b) Using `scipy.stats.binom.sf`, find the smallest number of correct predictions $k$ with
$\Pr(K \ge k) < 0.05$ under $H_0$. What accuracy is that?

(c) In words: what size of difference between checkpoints could this test *not* detect?
:::

## Why logistic regression is a natural classifier here

Section 5 covers logistic regression in detail, but one fact belongs here. Logistic regression models
the **log-odds** $\log \frac{\Pr(y=1\mid x)}{\Pr(y=0\mid x)}$ as a linear function. From the formula
above, with balanced classes, the true log-odds is

$$
\log \frac{\Pr(y=1 \mid x)}{\Pr(y=0 \mid x)} \;=\; \log \frac{q(x)}{p(x)}.
$$

So the classifier is trying to learn the **log density ratio** between the checkpoints: how many
times more likely the RL model is to write *this* story than the DPO model. A positive weight on a
word like *lighthouse* literally says that its presence makes the story more RL-like. This
connection comes back in Section 10, where averaging the log ratio gives a KL divergence.

\newpage

# Data: what you have, what you need, and confounds

## What's already in the repo

| File | Stories | Fields | Usable as a class? |
|:-----------------------|------:|:------------------------------------------|:---------------|
| `instruct-sft.jsonl` (in `new_stories/data/`) | 500 | `stage`, `prompt_index`, `sample_index`, `seed`, `story`, `finish_reason`, settings | Yes (warm-up) |
| `instruct-dpo.jsonl` (same folder) | 500 | same | Yes (warm-up) |
| old Think run (in `generated_stories/data/`) | 5,000 | only `i`, `content` | **No** (see below) |

The Instruct files were made by `helper_scripts/generate_stories.py`, which records every setting.
The old Think run records only the story text: no prompt, no sampling settings, no system prompt.

## The confound problem

::: {.concept title="A classifier detects every difference, including setup differences"}
The classifier learns $P$ vs. $Q$, where $P$ and $Q$ are the distributions of stories **as
generated**. They depend on the checkpoint *and* on the prompt, system prompt, sampling settings,
token cap and post-processing. If any of those differ between the two groups, the classifier will
happily learn them, and you'll read a setup difference as a training-stage difference.
:::

Concretely, three confounds are lying in wait:

1. **The old RL run.** One prompt ("write a story"), the old system prompt, and settings we don't
   know. Your Instruct runs added the line *"Each story should be about 500 words."* (that's the
   `--target-words 500` default in `generate_stories.py`). A classifier comparing them would learn
   story length and prompt effects.
2. **Branch.** Your SFT and DPO stories are *Instruct*; the final model you care about is *Think*.
   Comparing Instruct-DPO with Think-RL mixes up "branch" with "stage". You need **Think** SFT, DPO
   and RLVR, all generated identically.
3. **Truncation and failures.** Think models spend tokens reasoning before writing. With a
   4,096-token cap, some responses will end mid-reasoning (`finish_reason == "length"`) or produce
   an empty story. If one checkpoint fails much more often, the surviving stories are a *selected*
   sample (e.g. only the short-reasoning ones), which is another difference the classifier can
   learn.

::: {.pitfall title="Selection by failure"}
Always report, per checkpoint, how many responses finished (`stop`) vs. hit the cap (`length`).
For the main analysis, **drop a seed key from all three checkpoints if any one of them failed**,
so every checkpoint is evaluated on the same prompts and seeds. Then check that the results don't
change much if you keep everything.
:::

## The unit of analysis: the seed key

`generate_stories.py` sets the seed as

```python
seed = p * 100_000 + s  # same seed for the same (prompt, sample) on every checkpoint
```

so a pair `(prompt_index, sample_index)` names the **same request** on every checkpoint. Call it
the **seed key**. With 5 prompts $\times$ 100 samples there are 500 seed keys, each with one story
per checkpoint.

Could two checkpoints' stories with the same seed be *related*? DPO starts from the SFT weights, so
with the same random seed their first sampled tokens can coincide. I checked this on the Instruct
data:

| Pairing | Mean shared opening words | Share sharing $\ge 5$ opening words |
|---|---|---|
| Same seed key (SFT vs DPO) | 0.75 | 3.2% |
| Random pairs | 0.47 | 1.6% |

That's small, but it isn't zero. It's a reason to keep each seed key in **one** split
(Section 6).

::: {.exercise title="3.1 Reproduce the seed check"}
Load both Instruct files, index stories by `(prompt_index, sample_index)`, and compute the number
of identical leading words for each same-key pair. Then do the same for randomly shuffled pairs.
You should get numbers close to the table above.
:::

::: {.exercise title="3.2 Setup-check cell"}
Notebook 01, section 1, has a cell that checks both stages used the same repo revision, llama.cpp
build, chat template, sampling, system prompt and prompt counts. Copy it into notebook 02 and extend
it to three stages. Add: the `finish_reason` counts, the number of empty stories, and whether any
`story` contains `<think>`. This cell should be the first thing you run for every new dataset.
:::

\newpage

# Turning stories into features

A classifier needs each story as a vector of numbers $x \in \mathbb{R}^d$. The choice of features
decides **what kinds of differences** the classifier can see.

## Bag of words

The simplest choice: one dimension per word in a vocabulary $V$, with value equal to how often the
word occurs. Word order is ignored ("bag"). It's crude, but it's very strong for telling authors
or styles apart, and every dimension is a word you can read, which is what makes Section 9 possible.

**Tokenization.** scikit-learn's default token pattern is `r"(?u)\b\w\w+\b"`: runs of 2 or more
letters and digits, lowercased. Note what that drops: punctuation (em dashes, curly quotes),
single-character words ("I", "a"), and markdown (`**Title:**`). Sometimes those differences
are real style, and sometimes they're formatting artifacts. Section 9 has a character-n-gram
variant for checking.

**n-grams.** `ngram_range=(1, 2)` adds word pairs ("old lighthouse", "the keeper"). That's more
expressive, but the vocabulary is much bigger.

**Vocabulary size.** `min_df=2` drops words that appear in only one story (mostly typos and
one-off names). That shrinks the vocabulary and reduces overfitting.

## TF-IDF weighting

Raw counts let common words ("the", "and") dominate. **TF-IDF** downweights words that appear in
most documents. scikit-learn's default (with `smooth_idf=True`) is

$$
\operatorname{idf}(t) = \ln\frac{1 + N}{1 + \operatorname{df}(t)} + 1,
\qquad
x_{d,t} = \operatorname{tf}(t, d)\cdot \operatorname{idf}(t),
$$

where $N$ is the number of training documents and $\operatorname{df}(t)$ is how many contain $t$.
With `sublinear_tf=True`, $\operatorname{tf}$ becomes $1 + \ln(\text{count})$, so a word used 20
times counts for much less than 20 times a single use. Finally each row is scaled to unit length
($\lVert x_d \rVert_2 = 1$).

::: {.exercise title="4.1 TF-IDF by hand"}
Corpus: $d_1$ = "the lighthouse keeper", $d_2$ = "the baker", $d_3$ = "the lighthouse".
Using the formula above (raw tf, smooth idf, L2 normalization), compute the vector for $d_1$ over
the vocabulary (baker, keeper, lighthouse, the). Check your answer with
`TfidfVectorizer().fit_transform([...]).toarray()`.
:::

## Length: the feature you didn't ask for

L2 normalization removes *direct* length information, but length leaks in other ways: longer
stories use more distinct words, more rare words, more plot. In the Instruct pilot, **story length
alone** separates SFT from DPO with AUC $\approx 0.74$. Length is a real difference between
checkpoints, but it's an uninteresting one, and it can hide the interesting ones.

The fix used in notebook 01, section 6, carries over: **truncate every story to its first $L$ words**
(e.g. $L = 300$) before featurizing. Choose $L$ at or below the 10th-percentile length of the
shortest checkpoint, so almost every story is actually cut. Report results both ways.

## Ablations: deciding what the classifier is allowed to see

Building on the truncated text, make several versions of each story and train a separate
classifier on each:

| Variant | How | Question it answers |
|---|---|---|
| full | the story as is | How different are the checkpoints overall? |
| first-$L$ | first $L$ words | ...once length is removed? |
| Core-masked | `PATTERN.sub("CORE", text)` | ...without being able to *name* which Core word? |
| Core-removed | `PATTERN.sub("", text)` | ...without any Core information at all? |
| names-masked | replace mid-sentence capitalized words with `NAME` | ...beyond character and place names? |
| Core-only | 11 binary features: does word $w$ appear? | How much do the Core words *alone* explain? |
| length-only | 1 feature: word count | How much does length alone explain? |

::: {.repo}
`PATTERN` and `WORDS` come from `helper_scripts/core_rates.py`: the same whole-word,
case-insensitive regex that reproduced the paper's Table 5. Import it rather than retyping the word
list. For names, `mid_sentence_caps()` in notebook 01, section 8, already finds mid-sentence
capitalized words. Adapt it to *replace* them.
:::

::: {.exercise title="4.2 Masking: CORE vs nothing"}
Why might "Core-masked" (replace with the token `CORE`) and "Core-removed" (delete) give different
accuracies? Which one tests "do the checkpoints differ in *which* Core words they use" and which
tests "do they differ in *whether* they use Core words at all"?
:::

## Beyond bag of words (optional, later)

**Sentence embeddings** (e.g. the `sentence-transformers` library) map a story to a dense vector
that captures meaning, so "a keeper tended the light" and "she watched over the beacon" end up
close together. A logistic regression on embeddings catches *semantic* similarity that bag of words
misses, at the cost of interpretability. Do this only after the bag-of-words pipeline works end to
end. The rest of the pipeline (splits, tests) is identical: only the feature step changes.

\newpage

# The model: regularized logistic regression

## The model

For a feature vector $x \in \mathbb{R}^d$, logistic regression predicts

$$
\Pr(y = 1 \mid x) \;=\; \sigma(w^\top x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}},
$$

where $w \in \mathbb{R}^d$ (one weight per word) and $b$ is an intercept. Taking the log-odds:

$$
\log \frac{\Pr(y=1\mid x)}{\Pr(y=0\mid x)} \;=\; w^\top x + b \;=\; b + \sum_{t \in V} w_t\, x_t .
$$

Each word **adds** $w_t x_t$ to the log-odds. A positive $w_t$ pushes toward class 1 (say RL), a
negative one toward class 0. Equivalently, it **multiplies the odds** by $e^{w_t x_t}$.

::: {.exercise title="5.1 Reading a weight"}
Suppose "lighthouse" has weight $w = 3.0$ toward RL, and in a particular story its TF-IDF value is
$x = 0.12$. By what factor does this word multiply the odds that the story is from RL? If the
story's total log-odds is $0$ without it, what is $\Pr(\text{RL} \mid x)$ with it?
:::

## Fitting: maximum likelihood with a penalty

Given training stories $(x_i, y_i)_{i=1}^n$ and $p_i = \sigma(w^\top x_i + b)$, the negative
log-likelihood (the **log loss**, or cross-entropy) is

$$
\mathcal{L}(w, b) \;=\; -\sum_{i=1}^{n} \Bigl[\, y_i \log p_i + (1 - y_i) \log (1 - p_i) \Bigr].
$$

scikit-learn minimizes the **L2-regularized** version

$$
\min_{w,b}\;\; \tfrac12 \lVert w \rVert_2^2 \;+\; C \cdot \mathcal{L}(w, b).
$$

$C$ is the *inverse* regularization strength: a small $C$ means strong shrinkage of the weights
toward 0, and a large $C$ means the model fits the training data more closely.

::: {.exercise title="5.2 The gradient"}
Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, show that

$$
\nabla_w \mathcal{L} \;=\; \sum_{i=1}^n (p_i - y_i)\, x_i \;=\; X^\top (p - y).
$$

Interpret: each story pulls the weights in the direction of its own features, scaled by *how wrong*
the current prediction is. A story the model already gets right ($p_i \approx y_i$) barely moves
anything.
:::

**Convexity.** The Hessian of $\mathcal{L}$ is $X^\top S X$ with $S = \operatorname{diag}(p_i(1-p_i))$,
which is positive semidefinite, so the loss is convex. Adding $\tfrac12\lVert w\rVert^2$ makes it
*strictly* convex. There is exactly one best $w$, and the solver (`lbfgs`) finds it. No random
restarts, no training curves to babysit. That's a big reason to start with this model.

## Why regularization isn't optional here

In the Instruct pilot, truncated stories with `min_df=2` give a vocabulary of about **5,800 words**,
while a training set has about **800 stories**. With more features than data points ($d > n$), you
can almost always find a hyperplane that separates the training stories *perfectly*.

::: {.exercise title="5.3 What goes wrong without a penalty"}
Suppose the training data are perfectly separable: some $w_0$ has $w_0^\top x_i > 0$ exactly for
the $y_i = 1$ stories. Show that scaling $w = c\,w_0$ with $c \to \infty$ drives $\mathcal{L} \to 0$,
so the unpenalized maximum-likelihood estimate doesn't exist. What would that do to (a) the
predicted probabilities and (b) the interpretability of the weights?
:::

The penalty fixes this, and $C$ controls the **bias–variance trade-off**: too small and the model
can't express real differences (high bias); too large and it memorizes training quirks (high
variance). You choose $C$ on validation data (Section 6), never on test data.

## More than two classes

For SFT vs DPO vs RL together, **multinomial** logistic regression gives each class $k$ its own
weights and uses the softmax $\Pr(y=k\mid x) = e^{w_k^\top x}/\sum_j e^{w_j^\top x}$. Current
scikit-learn versions do this automatically when `y` has more than two labels. The **confusion
matrix** (which checkpoints get mistaken for which) is then the main output. Do the pairwise
comparisons first, though: they map directly onto the research question and are easier to test.

## Why not a neural network?

A fine-tuned transformer classifier would probably be more accurate (giving a tighter $\TV$ bound),
but it's harder to interpret, needs a GPU to train well, and adds a lot of tuning choices. For
this project, interpretability (Section 9) is the payoff, and logistic regression on TF-IDF is a
strong, standard baseline for text. If your mentor wants a tighter distance estimate later, the
same splits and tests work with any classifier.

\newpage

# Splitting the data

## Why three roles

| Split | Used for | Rule |
|---|---|---|
| **Training** | fitting $w$ | the model sees it |
| **Validation** | choosing $C$, `ngram_range`, `min_df`, $L$ | the *choices* see it |
| **Test** | the reported accuracy and tests | used **once**, at the end |

If you choose $C$ by looking at test accuracy, the test accuracy is no longer an honest estimate:
you've optimized it. With a small dataset, a good design is **a frozen test set plus
cross-validation on the rest**:

1. Hold out 20% of seed keys as the **test set** and save them to `results/stage_prediction/test_keys.json`.
2. On the remaining 80%, use **5-fold cross-validation**: train on 4 folds, validate on the 5th,
   rotate, and average. This is the "validation" step. Choose hyperparameters here.
3. Refit the chosen model on all of the 80%, and evaluate **once** on the test set.

## Splitting by seed key, stratified by prompt

Two constraints:

- **Group by seed key.** All stories with the same `(prompt_index, sample_index)` (one per
  checkpoint) go in the same split. That rules out the small same-seed leakage from Section 3, and
  it makes the design **paired**: every checkpoint is tested on the same 100 seed keys, so
  "SFT vs RL" and "DPO vs RL" are compared on identical RL test stories.
- **Stratify by prompt.** Each split gets the same share of each of the 5 prompts (20 test keys per
  prompt), so a prompt imbalance can't masquerade as a checkpoint difference.

::: {.exercise title="6.1 Write the split module"}
Complete this module. It's the one piece of reusable code for this part of the project.

```python
"""Load generated stories and make a frozen train/test split by seed key."""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "new_stories" / "data"
FIELDS = ("stage", "prompt_index", "sample_index", "seed",
          "story", "finish_reason")


def load_stage(name: str) -> list[dict]:
    # TODO: read DATA / f"{name}.jsonl"; keep only FIELDS from each record.
    ...


def seed_key(record: dict) -> tuple[int, int]:
    return record["prompt_index"], record["sample_index"]


def make_test_keys(keys, frac=0.2, seed=0) -> set[tuple[int, int]]:
    # TODO: group keys by prompt_index; for each prompt, shuffle its keys with
    # random.Random(seed) and take round(frac * len) of them for the test set.
    ...


def save_keys(keys, path: Path) -> None:
    # TODO: write sorted keys as a JSON list of [prompt_index, sample_index] pairs.
    ...


def load_keys(path: Path) -> set[tuple[int, int]]:
    # TODO: inverse of save_keys.
    ...
```

Then, in the notebook, write assertions: no key is in both train and test; every prompt has the
same number of test keys; every checkpoint has a story for every test key (after dropping failures).
:::

## Cross-validation that respects groups

For the inner folds, scikit-learn has `StratifiedGroupKFold`: it keeps groups (seed keys) intact
and balances the label $y$ across folds.

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
# groups: one integer per story, e.g. prompt_index * 1000 + sample_index
for train_idx, val_idx in cv.split(texts, y, groups):
    ...
```

## Fit the vectorizer inside each fold

::: {.pitfall title="Leakage through the vectorizer"}
`TfidfVectorizer` *learns* things from the text: the vocabulary (which words pass `min_df`) and the
idf weights (document frequencies). If you fit it on all stories and then split, the validation and
test stories have already shaped the features. The fix is to bundle vectorizer and classifier in
a `Pipeline`, so that `fit` on training data fits both and `predict` on held-out data only
*applies* them.
:::

::: {.exercise title="6.2"}
Explain concretely how fitting `min_df=2` on all data could leak test information. (Hint: think about
a rare place name that appears in exactly one training story and one test story.)
:::

## How many stories?

With 500 seed keys and a 20% test split, each checkpoint has 400 training and 100 test stories, and
a two-way comparison has 800 train / 200 test. From Exercise 2.3, that can detect accuracy above
roughly 56%, which is plenty when checkpoints differ as much as the Instruct ones do. A
**learning curve** (Section 7) tells you whether more stories would sharpen the answer.

\newpage

# Implementation in scikit-learn: warm-up on the Instruct data

Before the Think stories exist, build the whole pipeline on `instruct-sft` vs `instruct-dpo`. The
pipeline is identical. Only the file names change later.

## The pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 1)),
    LogisticRegression(C=1.0, max_iter=2000),
)
pipe.fit(train_texts, train_y)                # fits vocabulary, idf AND weights
proba = pipe.predict_proba(test_texts)[:, 1]  # P(class 1 | story)
pred = pipe.predict(test_texts)
```

`make_pipeline` names the steps after their classes in lowercase: `tfidfvectorizer` and
`logisticregression`. You'll need those names for the grid search.

## Choosing hyperparameters with grid search

```python
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

grid = {
    "logisticregression__C": [0.1, 0.3, 1, 3, 10, 30],
    "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
    "tfidfvectorizer__min_df": [2, 5],
}
search = GridSearchCV(
    pipe, grid, scoring="roc_auc",
    cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0),
)
search.fit(train_texts, train_y, groups=train_groups)
print(search.best_params_, round(search.best_score_, 3))
best = search.best_estimator_   # already refit on ALL training data
```

::: {.exercise title="7.1 Build the warm-up"}
In `notebooks/02_stage_prediction.ipynb`:

1. Load both Instruct files with your `stage_data.py`, make and save the test split, and run your
   setup-check cell.
2. Build the text variants from Section 4 (full, first-300, Core-masked, Core-removed,
   names-masked).
3. For each variant: grid search on the training keys, then report **test** accuracy and AUC
   (Section 8). Put everything in one DataFrame with one row per variant.
4. Add the two non-text baselines: **Core-only** (11 binary features) and **length-only**
   (word count; wrap it in `make_pipeline(StandardScaler(), LogisticRegression())`).
:::

::: {.checkpoint title="Expected ballpark for the Instruct warm-up"}
A quick pilot (5-fold grouped CV over all 1,000 stories, $C = 10$, unigrams) gave:

| Variant | Accuracy | AUC |
|---|---|---|
| full story | 0.90 | 0.97 |
| first 300 words | 0.84 | 0.93 |
| first 300, Core-masked | 0.84 | 0.93 |
| Core-only (11 features) | n/a | 0.59 |
| length-only | n/a | 0.74 |

Your test-set numbers will differ by a few points (different protocol, tuned $C$, 200 test stories),
but the *pattern* should hold. If you get 0.99+ or $\approx 0.50$ everywhere, suspect leakage or a
labelling bug.
:::

::: {.exercise title="7.2 Read the pattern"}
Using the ballpark table: (a) What does it mean that masking the Core words changes nothing?
(b) Core-only AUC is 0.59 even though the Core *rate* differs by 9 points (44% vs 53%). Why is a
9-point rate difference such a weak classifier? (Hint: what does the classifier do for the 47% of
DPO stories with no Core word?) (c) What do (a) and (b) together say about relying only on the
Core score?
:::

## Learning curve

```python
from sklearn.model_selection import learning_curve

sizes, train_sc, val_sc = learning_curve(
    best, train_texts, train_y, groups=train_groups,
    cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], scoring="roc_auc",
)
```

Plot the mean validation score against `sizes`. If it's still rising at the right edge, more stories
per checkpoint would help, which is useful evidence when you discuss sample size with your mentor.

\newpage

# Evaluation and statistical testing

## Metrics

**Accuracy** is the share of test stories classified correctly. It's easy to explain and maps
onto the $\TV$ bound.

**ROC AUC** doesn't need a threshold. If $s(x)$ is the classifier's score (e.g. the predicted
probability),

$$
\text{AUC} \;=\; \Pr\bigl(s(X^{+}) > s(X^{-})\bigr) + \tfrac12 \Pr\bigl(s(X^{+}) = s(X^{-})\bigr),
$$

where $X^+$ is a random class-1 story and $X^-$ a random class-0 story. In words: *pick one story
from each checkpoint; how often does the classifier rank them correctly?* 0.5 is chance and 1.0 is
perfect. It equals the Mann–Whitney $U$ statistic divided by $n^+ n^-$.

**Log loss** (the training objective, on test data) rewards well-calibrated probabilities, which
matters if you use the probabilities as RL-likeness scores (Section 9).

```python
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
```

## Confidence intervals

**Accuracy: Wilson interval.** Notebook 01, section 3, already defines `wilson(k, n)`. Reuse it with
$k$ = number correct and $n$ = test size.

**AUC (and anything else): grouped bootstrap.** Resample *seed keys* with replacement (not
individual stories, since each key contributes one story per class), recompute the metric on the
resampled test set, repeat $B = 2{,}000$ times, and take the 2.5th and 97.5th percentiles.

::: {.exercise title="8.1 Grouped bootstrap"}
Write `bootstrap_ci(metric, y, score, groups, B=2000, seed=0)`. For each replicate, draw
`len(unique_groups)` groups with replacement, gather *all* rows belonging to the drawn groups
(repeating rows for groups drawn twice), and compute `metric(y_b, score_b)`. Return the percentile
interval. Why resample groups rather than rows?
:::

## Is it better than chance? Two tests

**Binomial test** (quick): is the number correct on the test set surprising under $H_0$?

```python
from scipy.stats import binomtest
binomtest(k_correct, n_test, p=0.5, alternative="greater").pvalue
```

**Permutation test** (more general): shuffle the labels, rerun the *whole* procedure, and see how
often shuffled data does as well as the real labels. With $B$ shuffles and observed statistic
$T_{\text{obs}}$:

$$
p \;=\; \frac{1 + \#\{b : T_b \ge T_{\text{obs}}\}}{B + 1}.
$$

scikit-learn has `permutation_test_score`. Given `groups`, it permutes labels **within** groups.
Here a group is one seed key holding one story per checkpoint, so it swaps which checkpoint "wrote"
each story *within a pair*. That's exactly the right null for a paired design. (Check the docstring
for your installed version, since the `groups` behaviour is documented there.)

::: {.exercise title="8.2 The noise floor"}
Construct a comparison that **should** be at chance: split the 500 stories of *one* checkpoint
randomly into two fake classes (by seed key) and run the full pipeline. Report its accuracy and AUC.
This is your empirical "no difference" baseline, and every real comparison should be read against
it.
:::

## Comparing two distances

The central comparison is *is DPO closer to RL than SFT is?*, i.e.
$\text{AUC}(\text{SFT vs RL})$ vs $\text{AUC}(\text{DPO vs RL})$. Because both use the **same RL
test stories and the same seed keys**, use a **paired bootstrap**: in each replicate, draw one set of
seed keys and compute *both* AUCs on it, then record the difference. The percentile interval of the
differences is your CI. If it excludes 0, the two distances differ.

::: {.pitfall title="Many comparisons"}
Three checkpoint pairs $\times$ seven feature variants is 21 numbers. Some will look "significant"
by chance. Decide in advance which comparison is the main one (full vs first-$L$ text, SFT-vs-RL
compared with DPO-vs-RL) and treat the ablations as descriptive. Or adjust with Holm's method if
you report p-values for all of them.
:::

\newpage

# Interpreting what the classifier learned

## Top-weighted words

```python
vec, clf = best[0], best[-1]
words = vec.get_feature_names_out()
w = clf.coef_[0]                      # positive -> clf.classes_[1]
order = w.argsort()
print("most class-0:", list(words[order[:25]]))
print("most class-1:", list(words[order[-25:][::-1]]))
```

For the Instruct pilot (first 300 words), the SFT-leaning words included *village, streets, city,
Willowbrook, Lila, lived, park*, and the DPO-leaning words included *Marrow, rain, fog, gray, cold,
battered, boots, Clara, Silverwood, like, remembered*. So DPO stories are moodier and more
weather-heavy, with different favourite names and places. **None of these are Core words.**

::: {.pitfall title="Weights aren't effect sizes"}
With TF-IDF, a weight's size depends on the word's idf and on regularization. Correlated words
share weight between them. Treat the ranking as a list of *candidates*, then confirm them two
ways: (1) **stability**: refit on bootstrap resamples and count how often each word lands in the
top 25; (2) **direct counts**: compare document frequencies with the log-odds method below.
:::

## A direct check: weighted log-odds ratios

Monroe, Colaresi and Quinn (2008), "Fightin' Words", compare word use between two corpora with a
log-odds ratio smoothed by a Dirichlet prior. With counts $y_w^{i}$ of word $w$ in corpus $i$,
corpus sizes $n^{i}$, and prior counts $\alpha_w$ (e.g. proportional to the pooled frequency, with
total $\alpha_0$):

$$
\hat\delta_w = \log\frac{y_w^{i} + \alpha_w}{n^{i} + \alpha_0 - y_w^{i} - \alpha_w}
              - \log\frac{y_w^{j} + \alpha_w}{n^{j} + \alpha_0 - y_w^{j} - \alpha_w},
\qquad
\sigma^2(\hat\delta_w) \approx \frac{1}{y_w^{i} + \alpha_w} + \frac{1}{y_w^{j} + \alpha_w}.
$$

Rank words by the $z$-score $\hat\delta_w/\sigma(\hat\delta_w)$. Words that rank high both here and
in the classifier are robust findings.

::: {.exercise title="9.1"}
Implement the weighted log-odds with `CountVectorizer` on the training stories of two checkpoints.
Do the top 20 words by $z$-score overlap with the classifier's top 20?
:::

## Where do DPO stories sit between SFT and RL?

Train a classifier on **SFT vs RL** only (training keys). Then score the held-out stories of all
three checkpoints with its log-odds $f(x) = w^\top x + b$ (`best.decision_function(texts)`). Let
$\bar f_{S}, \bar f_{D}, \bar f_{R}$ be the mean scores of SFT, DPO and RL test stories. Define

$$
\text{position}(\text{DPO}) \;=\; \frac{\bar f_{D} - \bar f_{S}}{\bar f_{R} - \bar f_{S}} .
$$

A value of 0 means DPO stories look like SFT along the SFT-to-RL direction, and 1 means they look
like RL. Beyond 1 would mean DPO overshoots and RL partly undoes it. Bootstrap it over seed keys
for a CI.

::: {.exercise title="9.2 Position and its limits"}
(a) Compute the position.

(b) Plot histograms of $f(x)$ for the three checkpoints' test stories on one axis.

(c) This score only measures the one direction that separates SFT from RL. Describe a way DPO
could differ from *both* that this score would miss, and how the three-way multinomial
classifier's confusion matrix would reveal it.
:::

## Linking back to Core stories

Within each checkpoint, compare the RL-likeness score $f(x)$ of stories **with** vs **without** a
Core word (use `score_core` from `core_rates.py`). If Core stories score as more RL-like even in
the *Core-masked* model, then Core stories differ from other stories in more than just the 11
words: they're part of a broader "RL style" (lighthouse, storm, old keeper, and so on).

## Read the stories

Numbers aren't enough. Print the 5 most confidently "RL" SFT stories, the 5 most confidently "SFT"
RL stories, and a few misclassified ones. The project rules apply: read the story text only, keep
excerpts short in saved notebook outputs, and remember some generations may be explicit.

## Optional: punctuation and formatting

Rerun the first-$L$ variant with character n-grams:
`TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5)`. If accuracy jumps, the
checkpoints differ in punctuation or formatting (em dashes, quotes, markdown) that the word
tokenizer drops. Check which: some of it may be real style, and some may be output formatting.

\newpage

# A second view: cross-entropy and asymmetry (optional)

The classifier gives one symmetric distance. A different question is: **does RL cover less ground
than DPO?** That is the mode-collapse hypothesis, where RL keeps a few of DPO's favourite patterns
and drops the rest. Answering it needs an *asymmetric* measure.

## KL divergence

$$
\KL(Q \,\|\, P) \;=\; \sum_x q(x) \log\frac{q(x)}{p(x)} \;=\; \E_{x\sim Q}\Bigl[\log \tfrac{q(x)}{p(x)}\Bigr] \;\ge 0.
$$

Read it as: *how surprised is a model of $P$ by samples from $Q$*. It's large when $Q$ produces
things $P$ considers unlikely. It isn't symmetric.

::: {.exercise title="10.1 Mode collapse in miniature"}
Four story types. DPO, $P = (0.25, 0.25, 0.25, 0.25)$, writes each equally. RL,
$Q = (0.97, 0.01, 0.01, 0.01)$, has collapsed onto type 1. Compute $\KL(Q\|P)$ and $\KL(P\|Q)$ in nats.
Which is larger, and why? Translate into a statement about "a model of RL predicting DPO stories"
versus "a model of DPO predicting RL stories".
:::

## Estimating it with unigram language models

Fit a smoothed **unigram model** on each checkpoint's training stories:

$$
\hat P(w) \;=\; \frac{c_P(w) + \alpha}{N_P + \alpha\,\lvert V\rvert},
$$

with $c_P(w)$ the count of word $w$, $N_P$ the total words, $V$ the vocabulary from *all* training
text, and $\alpha \approx 0.1$. The **cross-entropy** of held-out text $T$ under model $\hat P$ is

$$
H(T; \hat P) \;=\; -\frac{1}{N_T} \sum_{w} c_T(w)\, \log_2 \hat P(w) \quad \text{(bits per word)}.
$$

Then estimate

$$
\widehat{\KL}(Q\|P) \;=\; H(T_Q; \hat P) - H(T_Q; \hat Q),
$$

where $T_Q$ is RL **test** text and both models were fit on **equal-sized** training sets (so
smoothing affects both equally). Swap the roles for $\widehat{\KL}(P\|Q)$.

::: {.exercise title="10.2 Implement and interpret"}
Use `CountVectorizer` (fit on all training text) to get count vectors. Compute all six directed
estimates among SFT, DPO and RL. Mode collapse from DPO to RL predicts $\widehat{\KL}(\text{DPO}\|\text{RL})
> \widehat{\KL}(\text{RL}\|\text{DPO})$. Is that what you see? What does a unigram model *fail*
to capture that a collapse in plots or characters might involve?
:::

::: {.concept title="Advanced: use the checkpoints themselves"}
The ideal $p(x)$ is the checkpoint's own probability of the story. llama.cpp can score text
under a model (see the `llama-perplexity` tool and the server's log-probability options in the
llama.cpp docs for build b11633). Scoring each checkpoint's stories under the *other* checkpoints
gives a direct, model-based version of this asymmetry. It's expensive and fiddly (chat template,
prompt tokens must be excluded), so it's a later extension to discuss with your mentor.
:::

\newpage

# The full Think experiment

## Generate the data

Only the final Think GGUF exists so far (`~/models/gguf/Olmo-3-7B-Think-Q8_0.gguf`). For SFT and DPO,
follow the steps in `CLAUDE.md`:

```bash
# 1. download (about 14.6 GB each), outside the project folder
uv run --no-project helper_scripts/download_models.py download sft dpo --output-dir ~/models

# 2. convert to Q8_0 with the Python 3.12 venv (NOT Homebrew's 3.14; see CLAUDE.md)
cd ~/src/llama.cpp
.venv/bin/python convert_hf_to_gguf.py ~/models/Olmo-3-7B-Think-SFT \
    --outtype q8_0 --outfile ~/models/gguf/Olmo-3-7B-Think-SFT-Q8_0.gguf

# 3. sanity check: must continue with "Paris"
./build/bin/llama-completion -m ~/models/gguf/Olmo-3-7B-Think-SFT-Q8_0.gguf \
    -n 12 --temp 0 -p "The capital of France is" -no-cnv
```

Then, for each of `sft`, `dpo`, `rlvr` with the matching GGUF loaded in `llama-server`:

```bash
uv run --no-project helper_scripts/generate_stories.py sft \
    --out new_stories/data/think-sft.jsonl --n-per-prompt 100 --target-words 0
```

::: {.pitfall title="Fix the settings once, before generating anything"}
`--target-words` defaults to **500**, which adds the "about 500 words" line. The project's standard
system prompt in `CLAUDE.md` doesn't have it. Pick one (ask your mentor) and use it for **all three**
Think checkpoints. Same for `--max-tokens`: for Think models the cap includes the reasoning,
so 4,096 may truncate many responses. `generate_stories.py` refuses to resume a file with different
settings, which protects you within a file but not *across* files.
:::

::: {.mentor}
1. Standard system prompt with or without the "about 500 words" line?
2. Token cap for Think models (4,096 vs larger), and how to treat truncated responses?
3. Are 500 stories per checkpoint enough, or should the learning curve decide?
4. Classifier-based distance, the cross-entropy view, or both? Any preferred representation
   (bag of words vs embeddings)?
:::

## The comparisons to run

| # | Comparison | Purpose |
|---|---|---|
| 1 | SFT vs RL | distance from the start of post-training to the final model |
| 2 | DPO vs RL | distance from DPO to the final model |
| 3 | SFT vs DPO | how much DPO changed things |
| 4 | RL vs RL (random halves) | noise floor (Exercise 8.2) |
| 5 | old 5,000-story RL vs new RL | **how much the setup alone changes stories** |

Comparison 5 is worth doing: it measures the confound from Section 3 directly. If the classifier
easily separates the old and new RL runs, that's concrete evidence that the old run can't be mixed
with new data (and a nice result to show your mentor).

## Results template

Fill in test-set values, each with a 95% CI:

| Comparison | full AUC | first-$L$ AUC | Core-masked AUC | names-masked AUC | Core-only AUC | length-only AUC |
|---|---|---|---|---|---|---|
| SFT vs RL | | | | | | |
| DPO vs RL | | | | | | |
| SFT vs DPO | | | | | | |
| RL vs RL | | | | | | |
| old RL vs new RL | | | | | | |

Plus: the paired-bootstrap CI for $\text{AUC}(\text{SFT,RL}) - \text{AUC}(\text{DPO,RL})$, the DPO
position from Section 9, the top-25 words per comparison, and (optionally) the six KL estimates.

## Reading the outcome

| SFT vs RL | DPO vs RL | Position of DPO | Interpretation |
|---|---|---|---|
| high | near chance | $\approx 1$ | The final style is set by **DPO**; RL changes little. |
| high | high | $\approx 0.5$ | Both DPO and RL contribute. |
| high | high | $\approx 0$ | **RL** produces the final style. |
| near chance | near chance | n/a | Style is fixed by SFT (or earlier). The Instruct Core rate (44% at SFT) hints this is possible. |

Then cross with the ablations: does the pattern survive Core masking? If the full classifier says
"DPO is where it happens" but the Core-only classifier says "RL", the 11 words and the overall style
evolve on different schedules, which is itself an interesting result.

::: {.checkpoint title="Definition of done"}
`stage_data.py` with the split functions and a saved `test_keys.json`; notebook 02 that runs top to
bottom from a fresh kernel; the setup-check table for all three checkpoints; the filled results
table with CIs; the noise-floor comparison; the top-words lists; a written interpretation section
(like notebook 01, section 10) dated and tied to the outputs.
:::

\newpage

# Pitfalls checklist {-}

Before you trust any number, check:

- [ ] Every checkpoint was generated with identical prompt set, system prompt, sampling, token cap,
      llama.cpp build and chat template (the setup-check cell).
- [ ] Only the **story** is featurized: no prompt, no reasoning, no `<think>` tags.
- [ ] Failed or truncated responses are counted per checkpoint, and keys are dropped consistently.
- [ ] The test split was made **by seed key**, stratified by prompt, saved, and never used for any
      choice.
- [ ] The vectorizer is inside the `Pipeline` (fit only on training folds).
- [ ] Hyperparameters were chosen by grouped CV on the training keys.
- [ ] Results are reported with and without length control.
- [ ] The noise-floor comparison is near 0.5.
- [ ] Top words were checked for stability and against direct counts.
- [ ] You read actual stories behind the numbers.

\newpage

# Appendix A: Answer key {-}

**2.1.** Summing the identity over $x$:
$\sum_x \max\{p,q\} = \tfrac12\sum_x (p + q) + \tfrac12\sum_x \lvert p-q\rvert = 1 + \TV(P,Q)$,
because each distribution sums to 1 and $\TV = \tfrac12\sum\lvert p-q\rvert$. Multiplying by $\tfrac12$
gives $\text{acc}^* = \tfrac12 + \tfrac12\TV$. Every classifier's accuracy is at most the Bayes
accuracy, so $\text{acc} \le \tfrac12 + \tfrac12\TV$, i.e. $\TV \ge 2\,\text{acc} - 1$.

**2.2.** $2(0.84) - 1 = 0.68$: at least 68% of the probability mass sits on stories where the two
checkpoints disagree about which is more likely. An accuracy of 0.55 gives $\TV \ge 0.10$, and only
if 0.55 is itself significantly above 0.5 (see 2.3).

**2.3.** (a) $\sqrt{0.25/200} \approx 0.035$. (b) $k = 113$ correct ($\Pr(K\ge 113) \approx 0.038$),
i.e. accuracy $\ge 56.5\%$. (c) Differences that the classifier can only exploit to about 55% or
less ($\TV \lesssim 0.1$ *as seen by this classifier*) can't be distinguished from chance with 200
test stories.

**3.1.** About 0.75 mean shared opening words for same-key pairs vs 0.47 for random pairs, with
3.2% vs 1.6% sharing at least five. Your shuffle seed may move the random-pair numbers slightly.

**4.1.** idf: baker $= \ln(4/2)+1 = 1.693$, keeper $= 1.693$, lighthouse $= \ln(4/3)+1 = 1.288$,
the $= \ln(4/4)+1 = 1$. Raw $d_1 = (0, 1.693, 1.288, 1)$, norm $\sqrt{2.867 + 1.658 + 1} = 2.351$,
so $d_1 = (0,\ 0.720,\ 0.548,\ 0.425)$.

**4.2.** Masking keeps a `CORE` token, so the classifier can still learn *how often* Core words occur,
but not *which* one. Removing deletes that signal entirely. If masked $\approx$ full, the checkpoints
don't differ much in *which* Core words they pick. If removed $<$ masked, *whether* they use Core
words at all helps separate them.

**5.1.** The odds multiply by $e^{3.0 \times 0.12} = e^{0.36} \approx 1.43$.
$\Pr = \sigma(0.36) \approx 0.589$.

**5.2.** $\partial \mathcal{L}/\partial z_i = -(y_i - p_i)$, since
$\frac{d}{dz}[y\log\sigma + (1-y)\log(1-\sigma)] = y(1-\sigma) - (1-y)\sigma = y - \sigma$. By the
chain rule with $z_i = w^\top x_i + b$, $\nabla_w \mathcal{L} = \sum_i (p_i - y_i)x_i$.

**5.3.** For a separating $w_0$, $c\,w_0^\top x_i \to +\infty$ for $y_i = 1$ and $\to -\infty$ for
$y_i = 0$, so every $p_i \to y_i$ and every log term $\to 0$. The infimum 0 is never reached, so no
finite minimizer exists. (a) Predicted probabilities go to exactly 0 or 1: overconfident, with
infinite log loss on any test mistake. (b) Weight magnitudes are arbitrary (they grow with $c$), so
their ranking reflects whatever happened to separate the training set, not stable effects.

**6.2.** With `min_df=2` fitted on all data, a place name occurring once in a training story and once
in a test story *passes* the threshold and gets a feature. The training story teaches the model that
the name signals its class, and the test story is then "predicted" from a feature that exists only
because the test story was looked at. With the vectorizer fitted on training data only, the name
appears once, is dropped, and can't leak.

**7.2.** (a) The classifier doesn't need the identity of the Core words: the checkpoints differ in
many other words. (b) A binary feature that's present in 53% vs 44% of stories barely separates
individual stories. Most stories from both checkpoints look the same on it, including all the
no-Core ones. A rate difference that is significant across 500 stories can still be a weak
predictor of any *single* story. (c) The Core rate captures a real but small part of how the
checkpoints differ. Most of the difference lies elsewhere.

**8.1.** Rows sharing a seed key aren't independent (each key contributes one story per checkpoint,
and the pair is the unit that was randomized into the test set). Resampling rows would understate
the variance and break the pairing.

**8.2.** Expect accuracy within about $0.5 \pm 0.07$ and AUC near 0.5. A clearly higher value means a
bug, e.g. a leaked feature such as a key-dependent artifact.

**9.2(c).** DPO could move *off* the SFT-to-RL line, e.g. adopt a style neither SFT nor RL has, while
projecting to the middle. The multinomial confusion matrix would show DPO stories rarely mistaken
for either SFT or RL, even though the projection says "halfway".

**10.1.** $\KL(Q\|P) = 0.97\ln 3.88 + 3(0.01)\ln 0.04 \approx 1.219$ nats.
$\KL(P\|Q) = 0.25\ln(0.25/0.97) + 0.75\ln 25 \approx 2.075$ nats. $\KL(P\|Q)$ is larger because $P$
(DPO) puts lots of mass on types that $Q$ (RL) considers very unlikely. A model of RL is badly
surprised by DPO's variety, while a model of DPO is only mildly surprised by RL's repetitiveness.
Under collapse, *RL predicts DPO poorly, but DPO predicts RL reasonably well*.

**10.2.** A unigram model sees only word frequencies. It can't capture repeated plot structures,
character roles or story arcs that reuse different words, so it can miss a collapse that's
structural rather than lexical.

# Appendix B: scikit-learn cheat sheet {-}

| Task | API |
|---|---|
| Text to TF-IDF | `TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1,2))` |
| Text to counts | `CountVectorizer(min_df=2)` |
| Classifier | `LogisticRegression(C=1.0, max_iter=2000)` |
| Bundle steps | `make_pipeline(vectorizer, classifier)` |
| Grouped CV | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` |
| Tune | `GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv).fit(X, y, groups=g)` |
| CV score | `cross_val_score(pipe, X, y, groups=g, cv=cv, scoring="roc_auc")` |
| Probabilities | `pipe.predict_proba(X)[:, 1]` |
| Log-odds score | `pipe.decision_function(X)` |
| Vocabulary | `pipe[0].get_feature_names_out()` |
| Weights | `pipe[-1].coef_[0]` (positive means `pipe[-1].classes_[1]`) |
| Metrics | `accuracy_score`, `roc_auc_score`, `log_loss`, `confusion_matrix` |
| Permutation test | `permutation_test_score(pipe, X, y, groups=g, cv=cv, n_permutations=200)` |
| Learning curve | `learning_curve(pipe, X, y, groups=g, cv=cv, train_sizes=[...])` |
| Scale numeric features | `StandardScaler()` |

# Appendix C: Further reading {-}

- **Lopez-Paz, D. and Oquab, M. (2017).** *Revisiting Classifier Two-Sample Tests.* ICLR. The basis
  of Section 2.
- **Jurafsky, D. and Martin, J. H.** *Speech and Language Processing* (3rd ed. draft), chapters on
  logistic regression and n-gram language models. Clear derivations of Sections 5 and 10.
- **Hastie, T., Tibshirani, R. and Friedman, J.** *The Elements of Statistical Learning*, ch. 7
  (model assessment and cross-validation, including the "wrong way" to do CV).
- **Monroe, B. L., Colaresi, M. P. and Quinn, K. M. (2008).** *Fightin' Words.* Political Analysis.
  The weighted log-odds method in Section 9.
- **Kirk, R. et al. (2024).** *Understanding the Effects of RLHF on LLM Generalisation and Diversity.*
  ICLR. Already on your reading list, and directly relevant to the mode-collapse question.
- **scikit-learn user guide:** "Working with text data", "Cross-validation: evaluating estimator
  performance" (especially the section on group-aware splitters), and "Permutation test score".
- **Hamilton, S. and Mimno, D. (2026).** *Elias in the Lighthouse, Again?* Re-read §3–4 with this
  framing in mind: which of their measurements could be recast as prediction tasks?
