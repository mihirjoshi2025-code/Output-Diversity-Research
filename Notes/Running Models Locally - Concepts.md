---
tags: [setup, llama-cpp, local-models, quantization, sampling, week1]
---

# Running models locally: concepts

> [!summary] The whole note in 5 lines
> - A model is a **big pile of numbers** (weights). OLMo 7B has ~7 billion of them.
> - We **shrink** them from 16-bit to 8-bit (**Q8_0**) so they fit nicely on the laptop: 14.6 GB → 8 GB.
> - **llama.cpp** is the program that runs the shrunk model. **GGUF** is its file format.
> - Speed is limited by how fast the Mac can **read the 8 GB** for every word (~11 words/sec).
> - **Dice settings** (temperature etc.) change how varied the stories are, so they must be **identical for every checkpoint**.

See also: [[Parallel Generation - 4 Stories at Once]] · [[Servers and APIs - How llama-server Works]] · [[Shell and Command Syntax - Cheat Sheet]] · [[Reading List - Running Models Locally]]

---

## The 4 steps we follow

```
1. DOWNLOAD          2. CONVERT              3. SERVE              4. GENERATE
Hugging Face   ───►  one .gguf file   ───►   llama-server   ◄───►  our script
14.6 GB, 16-bit      8 GB, 8-bit (Q8_0)      model in memory       saves stories
```

---

## 1. What's in a model download

A model = **instructions for the math** + **the learned numbers** (weights / [[Parameters]]).

| File | Plain meaning |
|---|---|
| `*.safetensors` | The numbers themselves (the big files) |
| `config.json` | The model's shape: 32 layers, etc. |
| `tokenizer.json` | How to chop text into word-pieces |
| `chat_template.jinja` | How to format a conversation for this model |

**Revision** = a version ID (commit hash) for the download. We pin it so "the SFT model" always means the exact same numbers.

---

## 2. Shrinking: precision and quantization

Each number can be stored with more or fewer bits. Fewer bits = smaller file, slightly less exact.

| Format | Size of OLMo 7B | Used for |
|---|---|---|
| 16-bit (BF16) | 14.6 GB | What Ai2 publishes |
| **8-bit (Q8_0)** | **8 GB** | **What we use.** Almost no quality loss. |
| 4-bit (Q4) | ~4.5 GB | Smaller machines; noticeably worse |

**How Q8_0 works:** numbers are grouped in blocks of 32. Each block stores one "scale" plus 32 small whole numbers. Real value ≈ scale × small number.

**Why Q8_0:** the paper's OLMo run used it, and every checkpoint must use the same format so differences come from training, not file format.

> [!warning] Our bug
> Our first conversion flipped the sign on half the blocks (a numpy + Python 3.14 bug), so the model output `@@@@`. **Always run the "Paris" check after converting.**

---

## 3. llama.cpp and GGUF

- **llama.cpp**: a fast program for running models on normal computers.
- **GGUF**: its single-file format. One `.gguf` holds the numbers *and* the tokenizer and chat template.

| Tool | Job |
|---|---|
| `convert_hf_to_gguf.py` | Download folder → `.gguf` |
| `llama-completion` | Quick test in the terminal (the "Paris" check) |
| `llama-server` | Runs the model and answers requests |

---

## 4. Memory and speed on the M4

**Memory (24 GB total, shared by CPU and GPU):**
- Model: ~8 GB
- **KV cache** (each story's memory of what it's written so far): about **0.5 MB per token**
  - `-c 8192` → ~4 GB
  - `-c 16384` → ~8 GB
- Plus macOS and every open app. If it all doesn't fit, everything gets very slow.

**Speed:**
- Reading the prompt: fast (done all at once).
- Writing the answer: **one token at a time**, and each token means reading all 8 GB of numbers.
- The M4 reads about 120 GB/s → at most ~15 tokens/s. We get ~11.
- `--parallel 4` gets more out of each read (see [[Parallel Generation - 4 Stories at Once]]).

---

## 5. Tokens and chat templates

**Token** = a word-piece. Roughly ¾ of a word. 500 words ≈ 700 tokens. All limits and speeds are counted in tokens.

**Chat template** = the exact text format the model was trained on. Our messages get turned into something like:
```
<|im_start|>system
You are a fiction writer. ...<|im_end|>
<|im_start|>user
Write a story.<|im_end|>
<|im_start|>assistant
```
The `<|...|>` bits are **special tokens**: markers the model learned. When the model writes `<|im_end|>`, it's done (`finish_reason: stop`).

- `--jinja` = "use the template stored in the model file."
- Wrong template = the model sees a format it never trained on = weird behaviour.
- The Instruct models' built-in template has a bug in llama.cpp, so we use `chat_templates/olmo3-instruct-fixed.jinja` (gives the identical text).

**Think vs Instruct:** Think models write `<think>…</think>` reasoning before the answer. Instruct models answer directly. Our script saves the two parts separately.

---

## 6. Dice settings (sampling)

At every step the model has odds for every possible next token. **Sampling** = rolling weighted dice to pick one.

| Setting | Plain meaning | Ours |
|---|---|---|
| **temperature** | How much the dice favour the top choice. Lower = safer, more repetitive. 0 = no dice at all. | 0.6 |
| **top_k** | Only roll among the top k choices | 40 |
| **top_p** | Only roll among the top choices that add up to p% | 0.95 |
| **min_p** | Drop any choice far less likely than the top one | 0.05 |
| **seed** | The dice's starting number. Same seed → same rolls. | different per story |
| **max_tokens** | Longest allowed answer | 4,096 |

**Why it matters for us:** lower temperature → the model picks its favourite (Elara, lighthouse…) even more often. Different settings across checkpoints would ruin the comparison.

**Seeds aren't a perfect guarantee:** a different llama.cpp version or `--parallel` setting can change results slightly. So we record *everything*, not just the seed.

**finish_reason:** `stop` = finished on its own ✅ · `length` = cut off ⚠️ (count these; don't hide them).

---

## 7. Python environments (why `source .venv/bin/activate`)

- A **venv** is a private box of Python packages for one project, so projects don't break each other.
- `source .venv/bin/activate` = step into the box. `deactivate` = step out.
- **uv** is the tool that makes these boxes and installs packages.
- The llama.cpp converter's box must use **Python 3.12** (3.14 caused our `@@@@` bug).

---

## 8. Other tools you'll hear about

| Tool | One line | Why we don't use it |
|---|---|---|
| Ollama, LM Studio, Llama app | Friendly apps on top of llama.cpp | Hide settings; hard to record exactly what ran |
| MLX | Apple's own fast framework | Different math from the paper's run |
| vLLM | Fast server for big NVIDIA GPUs | Needs a GPU machine |
| Transformers | Python library, runs the full 16-bit model | Slow on a laptop |

---

## 9. Checklist: what every run must record

Model + revision · the GGUF file loaded · llama.cpp version · chat template · system + user prompt · every dice setting + seed · `max_tokens`, `-c`, `--parallel` · raw output, reasoning, story, `finish_reason`.

`helper_scripts/generate_stories.py` saves all of these for every story.
