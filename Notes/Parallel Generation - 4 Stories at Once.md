---
tags: [setup, llama-cpp, sampling, explainer]
---

# Parallel generation: how 4 stories get written at once

> [!summary] The whole note in 4 lines
> - The server writes **4 different stories at the same time** (`--parallel 4`).
> - They share the **model** (the slow part to load), not the **story so far** or the **dice**.
> - Each story rolls its own dice (its own **seed**), so they go different ways from the first word.
> - Same prompt + same seed = same story. That's why every story gets a different seed.

---

## Step 1: How the model writes *one* story

The model writes **one word-piece ([[Token|token]]) at a time**. Each step:

1. It reads **the story so far**.
2. It makes a **list of guesses** for the next word, with odds:
   ```
   "The"   40%
   "Once"  25%
   "In"    20%
   ...thousands more, each tiny
   ```
3. It **rolls dice** to pick one, weighted by the odds. "The" wins most often, but not always.
4. It adds that word to the story and goes back to step 1.

That repeats about 700 times for a 500-word story.

> [!note] Rolling dice = "sampling"
> The dice are why you get a different story every time. **Temperature** (we use 0.6) controls how much the dice favour the top choice. At temperature 0 there are no dice: it always picks the top word, so the same prompt always gives the same story.

---

## Step 2: What's slow about it

To make **each** guess list, the computer has to read through the **entire model**: about 8 GB of numbers. That reading is the slow part. It's why we get only ~11 words per second.

Think of it like this:
- The **model** is a giant recipe book in a warehouse across town.
- To cook **one bite**, the chef has to drive to the warehouse and flip through the whole book.
- That drive is what takes all the time. The cooking itself is quick.

---

## Step 3: The trick: one trip, four orders

If the chef is driving to the warehouse anyway, they might as well bring **4 orders** and work on all of them while they're there.

That's `--parallel 4`:
- **One trip** through the model per step.
- That one trip makes the **next word for 4 different stories**.
- Each story moves a bit slower, but 4 stories finish **much sooner than writing them one by one**, so total output goes way up. (The script's "min left" estimate shows the real speed.)

---

## Step 4: Why the 4 stories don't come out identical

They share the **recipe book** (the model). They **don't** share anything else.

| | Shared by all 4? |
|---|---|
| The model (8 GB of numbers) | ✅ Yes: that's the whole point |
| The story so far | ❌ No: each story has its own notepad ([[KV Cache]]) and only reads its own |
| The dice | ❌ No: each story has its own dice, set up with its own **seed** |
| The guess list | ❌ No: 4 separate lists come out, one per story |

### Walking through it
All 4 slots get the **same prompt**, "Write a story.":

```
Step 1 (same odds for all 4, because they've seen the same thing)
  story A rolls → "The"
  story B rolls → "Once"
  story C rolls → "In"
  story D rolls → "The"

Step 2 (now they've seen different things, so the odds differ)
  story A: "The"  → "lighthouse"
  story B: "Once" → "upon"
  story C: "In"   → "the"
  story D: "The"  → "city"

...and they drift further apart with every word.
```

Even A and D, which both started with "The", split at step 2, because their dice are different.

In our 5-story test, we got: a town called Larkspur, a clock tower in Elden Hollow, a city at dawn, a girl named Lila, and a girl named Elara in Willowbrook. All different.

---

## Step 5: When you *would* get copies

| Situation | Result |
|---|---|
| Same prompt, **different seeds** (what we do) | Different stories ✅ |
| Same prompt, **same seed** | The **same story**, word for word (whether parallel or not) |
| **Temperature 0** (no dice) | The **same story** every time |

Our script gives every story its own seed: `prompt number × 100,000 + story number`. Story #17 of "Tell me a story." gets the **same seed on SFT and on DPO**, which keeps the comparison fair.

---

## Step 6: "But the stories still feel samey…"

That's **not** parallel generation. That's the model.

If the model's guess list says "Elara" has 60% odds whenever it names a girl, then most dice rolls will land on Elara, no matter how many separate stories you run. Separate dice don't help when the dice are loaded.

**That's exactly what our project measures:** how loaded the dice are at each training stage (SFT vs DPO vs final).

---

## Step 7: Which order the stories get written in

- The script makes a to-do list of 500 stories: "Write a story." #0–99, then "Write me a story." #0–99, and so on.
- 4 workers each grab the next story off the list. When one finishes, it grabs another.
- So it works through the prompts **one after another**, 4 stories at a time. A prompt is never split or merged.
- If you stop halfway, the early prompts will be done and the later ones empty. Rerunning the same command fills in the rest.

---

## Words used here
- **Token:** a word-piece. 500 words ≈ 700 tokens.
- **Sampling:** rolling weighted dice to pick the next token.
- **[[Seed]]:** the starting number for the dice. Same seed → same rolls.
- **Temperature:** how much the dice favour the top choice.
- **Slot:** one of the server's 4 "lanes" for a story.
- **KV cache:** each story's private memory of what it's written so far.

Related: [[Running Models Locally - Concepts]] · [[Servers and APIs - How llama-server Works]]
