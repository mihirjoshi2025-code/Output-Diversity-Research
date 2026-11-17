# Project 1: When do OLMo 3 models start writing "Core" stories?

**Mentors / paper:** Sil Hamilton & David Mimno, *Elias in the Lighthouse, Again? Diagnosing Low Diversity in LLM Stories* (2026)
**Status:** Phase 0 (week 1): background and setup
**Scope of this document:** Week 1 only. Later phases will be added once setup is done and the first numbers are in.

---

## 1. The question

Give **the same prompts with the same settings** to three checkpoints of OLMo 3 7B Think: after SFT, after DPO, and after RLVR (the final model). At which stage do "Core" stories (Elias / Mara / Elara, keeper / baker / mayor / clockmaker / fisherman / librarian / conductor, lighthouse) appear or become much more common?

**Why it matters.** The paper finds that the share of Core stories in OLMo's *training data* rises only modestly across stages (SFT 3.0% → DPO 8.0% → RL 10.9%, Table 2). The final model writes Core stories **73% of the time** (in the existing 5,000-story OLMo run). If one stage produces a much bigger jump in the model's outputs than in its data, the training method is likely amplifying the pattern, not just copying it. That sets up Question 3 (do Core stories get extra weight during DPO?).

---

## 2. Background: what is OLMo 3?

**OLMo** ("Open Language Model") is a family of LLMs from the **Allen Institute for AI (Ai2)**. Most companies release only the final weights, if that. Ai2 releases **everything**: the weights at each training stage, the full training data, the training code, and the logs. **OLMo 3** (released late 2025) comes in 7B and 32B sizes. We use the **7B** models.

That openness is the reason this project uses OLMo. It's the only model where we can (a) download a checkpoint from *each* stage of training and (b) read the exact data used at each stage. You can't do either with Claude, GPT or Gemini.

### How an LLM like OLMo 3 Think is trained

Training happens in steps. Each step starts from the previous step's model:

```
Pretraining ──► Base model ──► SFT ──► DPO ──► RLVR ──► OLMo-3-7B-Think (final)
 (Dolma 3)                   (Dolci-Think-SFT) (Dolci-Think-DPO) (Dolci-Think-RL)
```

| Stage | What the model learns from | What it does to the model | Checkpoint we use |
|---|---|---|---|
| **Pretraining** | Trillions of words of web text, books, code, papers | Learns language and world knowledge by predicting the next word. Produces a **base model** that autocompletes text but doesn't follow instructions or chat. | *(not in scope yet)* |
| **SFT** (supervised fine-tuning) | ~Millions of example conversations: a prompt plus an ideal response (for Think models, the response includes a written-out `<think>…</think>` reasoning trace) | **Imitation.** The model learns to copy the style of good responses, including how to answer requests like "write a story." | `Olmo-3-7B-Think-SFT` |
| **DPO** (direct preference optimization) | Pairs of responses to the same prompt: one **chosen** (better) and one **rejected** (worse) | **Preference.** Raises the probability of the chosen response *relative to* the rejected one. This isn't imitation: it can strongly favour whatever chosen responses tend to share. | `Olmo-3-7B-Think-DPO` |
| **RLVR** (reinforcement learning with verifiable rewards) | Prompts where the answer can be checked automatically (math answers, code tests, instruction constraints like "use exactly 3 paragraphs"), plus some judged chat | **Reward.** The model generates answers, gets a reward when they check out, and shifts toward rewarded behaviour. | `Olmo-3-7B-Think` (final) |

The training data for the post-training stages is called **Dolci** (`Dolci-Think-SFT`, `Dolci-Think-DPO`, `Dolci-Think-RL`). Our `training_stories/` folder has the story-like extracts from these datasets.

**Think vs. Instruct.** OLMo 3 has two post-training branches from the same base model. **Think** models write a long reasoning trace in `<think>…</think>` before answering. **Instruct** models answer directly. We study **Think**. The `dolci-instruct-*` files belong to the other branch, so ignore them for now.

---

## 3. What's in this directory

```
AI Research - Cornell/
├── PROJECT_SCOPE.md          ← this file
├── helper_scripts/
│   ├── core_rates.py         ← counts the 11 Core words in stories
│   └── download_models.py    ← downloads the SFT / DPO / RLVR checkpoints (pinned revisions)
├── generated_stories/
│   ├── README.pdf            ← notes on how the stories were generated (read this!)
│   ├── prompts.docx          ← the 5 prompts
│   ├── manifest.json
│   └── data/                 ← 4 × 5,000 stories (Claude, Gemini, GPT, OLMo) + story_metadata.csv
└── training_stories/
    ├── README.pdf            ← notes on the Dolci extracts (read this!)
    ├── manifest.json
    └── data/                 ← Dolci Think/Instruct SFT, DPO, RL extracts + released ID list
```

### Key facts from the data notes

- **The existing OLMo stories were generated locally with llama.cpp** using a **Q8_0 GGUF** of the final `Olmo-3-7B-Think`. That's why your mentor pointed you to llama.cpp. Using the same setup for the SFT and DPO checkpoints keeps the comparison fair.
- The saved OLMo script used:
  - **System prompt** (use this exactly):
    > You are a fiction writer.
    > When the user asks for a story, respond with only the story text. Do not ask clarifying questions. Do not add introductions, markdown separators, bullet lists, or follow-up suggestions.
  - **Max output: 4,096 tokens**, with everything up to `</think>` removed before saving.
  - **A single repeated "write a story" prompt**, *not* the 5 prompts. For the new comparison, use **all 5 prompts** for every checkpoint.
- Old rows don't record seeds, finish reasons or token counts. **New runs must save:** prompt, checkpoint + revision, all sampling settings, the raw reasoning, the final story, and the finish reason.
- **Score only the story** (never the prompt or the reasoning).
- Some training extracts contain explicit content. Some aren't stories at all.

### Checkpoints (pinned in `download_models.py`)

| Stage | Hugging Face repo | Revision |
|---|---|---|
| sft | `allenai/Olmo-3-7B-Think-SFT` | `6ff8575…` |
| dpo | `allenai/Olmo-3-7B-Think-DPO` | `7b18bf9…` |
| rlvr | `allenai/Olmo-3-7B-Think` | `d97e442…` |
| instruct-sft | `allenai/Olmo-3-7B-Instruct-SFT` | `e1452fc…` |
| instruct-dpo | `allenai/Olmo-3-7B-Instruct-DPO` | `b33130b…` |

Each download is about **14.6 GB**.

---

## 4. Week 1 plan

### Day 1–2: Reading

1. **The paper.** Read it closely, especially §3–4, Table 2 (Core rate by training stage) and Table 5 (Core words per model).
2. **The two data READMEs** in `generated_stories/` and `training_stories/`.
3. **OLMo 3 tech report** (Ai2): the post-training sections for the Think models. Find out what data each stage used, how chosen/rejected DPO pairs were built, and which tasks RLVR rewards. Write a few notes on each stage.
4. **Background on alignment**, skimming is fine:
   - Ouyang et al. 2022 (InstructGPT): the SFT → preference-training pipeline.
   - Rafailov et al. 2023 (DPO): read only the intro and the main equation.
   - Kirk et al. 2024: how RLHF reduces output diversity. This one is closest to our question.
5. **llama.cpp README**: the sections on building, converting models (`convert_hf_to_gguf.py`) and `llama-server`.

### Day 2–3: Environment setup

Your machine is an Apple M4 with 24 GB of memory and about 144 GB of free disk. `uv`, `git` and Homebrew are already installed.

**Step 1: Tidy the folders (optional but recommended).** The READMEs and helper commands assume `helper_scripts/`, `generated_stories/` and `training_stories/`. Spaces and `(1)` in the names make shell commands awkward. Rename them, or remember to quote paths.

**Step 2: Start version control for your own code.**
```bash
cd ~/"AI Research - Cornell"
git init
# add a .gitignore that excludes data/, models, *.gguf, *.safetensors
```

**Step 3: Check the Core scorer against the paper.**
```bash
uv run --no-project helper_scripts/core_rates.py jsonl generated_stories/data > core_rates.json
```
The OLMo file should give exactly the OLMo column of Table 5. (Checked: it matches. 73.1% of OLMo stories contain a Core word. Elara 6,625.5 PPM, Elias 893.2, lighthouse 1,958.7, keeper 713.5.) The OLMo stories already have the reasoning removed, so `--strip-thinking` isn't needed for this file.

**Step 4: Build llama.cpp from source.** Building from source (instead of `brew install llama.cpp`) keeps the conversion script and the binaries at the same version, and gives you a commit hash to record.
```bash
brew install cmake
mkdir -p ~/src && git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
cd ~/src/llama.cpp
cmake -B build                          # Metal (Apple GPU) support is on by default on macOS
cmake --build build --config Release -j
git rev-parse HEAD                      # ← write this commit hash in your notes
```

**Step 5: Set up the Python environment for converting models.**
```bash
cd ~/src/llama.cpp
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements/requirements-convert_hf_to_gguf.txt
```

**Step 6: Download one checkpoint first (the final RLVR model).** Keep models **outside** this project folder:
```bash
cd ~/"AI Research - Cornell"
uv run --no-project helper_scripts/download_models.py download rlvr --output-dir ~/models
```

**Step 7: Convert it to a Q8_0 GGUF** (the same format the paper's OLMo run used):
```bash
cd ~/src/llama.cpp && source .venv/bin/activate
mkdir -p ~/models/gguf
python convert_hf_to_gguf.py ~/models/Olmo-3-7B-Think \
    --outtype q8_0 --outfile ~/models/gguf/Olmo-3-7B-Think-Q8_0.gguf
```
The result is about 8 GB. Once the GGUF works, you can delete the 14.6 GB safetensors folder to save disk. The download script can fetch it again later if needed.

**Step 8: Smoke test with `llama-server`.** Your generation scripts will call this same server later.
```bash
~/src/llama.cpp/build/bin/llama-server \
    -m ~/models/gguf/Olmo-3-7B-Think-Q8_0.gguf --jinja -c 8192 --port 8080
```
In another terminal:
```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "You are a fiction writer.\nWhen the user asks for a story, respond with only the story text. Do not ask clarifying questions. Do not add introductions, markdown separators, bullet lists, or follow-up suggestions."},
    {"role": "user", "content": "Write a story."}
  ],
  "max_tokens": 4096
}'
```
Check:
- [ ] The response contains a story. Is the reasoning inline in `<think>…</think>`, or in a separate `reasoning_content` field? (This depends on llama-server's reasoning settings. You need to save the reasoning either way.)
- [ ] `finish_reason` is `stop`, not `length` (`length` means it hit the 4,096-token cap).
- [ ] Note the speed (`timings.predicted_per_second` in the response) and which sampling defaults the server used (temperature, top_p, top_k, min_p).

**Step 9: Estimate how long generation will take.** Run 5–10 stories and fill in:

| | Value |
|---|---|
| Tokens per second | |
| Average tokens per response (reasoning + story) | |
| Minutes per story | |
| Hours for 1,000 stories × 3 checkpoints | |

At typical laptop speeds this may come to **days** of generation. If so, bring the numbers to your mentor and ask about a GPU machine (llama.cpp also runs on NVIDIA GPUs). `llama-server --parallel N` (several requests at once) can also improve throughput, so try it.

**Step 10: Load the SFT and DPO checkpoints (Instruct branch).** Use the **Instruct** versions, not the Think ones. Do one model at a time so only one 14.6 GB download sits on disk.

*a. Download* (from the project folder):
```bash
cd ~/"AI Research - Cornell"
uv run --no-project helper_scripts/download_models.py download instruct-sft --output-dir ~/models
uv run --no-project helper_scripts/download_models.py download instruct-dpo --output-dir ~/models
```

*b. Convert to Q8_0.* The venv must be the Python 3.12 one (see `CLAUDE.md`, "Conversion gotcha"):
```bash
cd ~/src/llama.cpp && source .venv/bin/activate
python --version                        # must say 3.12.x
python convert_hf_to_gguf.py ~/models/Olmo-3-7B-Instruct-SFT \
    --outtype q8_0 --outfile ~/models/gguf/Olmo-3-7B-Instruct-SFT-Q8_0.gguf
python convert_hf_to_gguf.py ~/models/Olmo-3-7B-Instruct-DPO \
    --outtype q8_0 --outfile ~/models/gguf/Olmo-3-7B-Instruct-DPO-Q8_0.gguf
deactivate
```

*c. Sanity-check each GGUF* (should continue with "Paris"; garbage like `@@@@` means a bad conversion):
```bash
cd ~/src/llama.cpp
./build/bin/llama-completion -m ~/models/gguf/Olmo-3-7B-Instruct-SFT-Q8_0.gguf -n 12 --temp 0 -p "The capital of France is" -no-cnv
./build/bin/llama-completion -m ~/models/gguf/Olmo-3-7B-Instruct-DPO-Q8_0.gguf -n 12 --temp 0 -p "The capital of France is" -no-cnv
```
Once a GGUF passes, you can delete its safetensors folder (`~/models/Olmo-3-7B-Instruct-SFT`, etc.).

*d. Serve one model at a time* (stop the running server with Ctrl-C first; only one can use port 8080).

**Run these from the project folder** (`cd ~/"AI Research - Cornell"`), because `--chat-template-file` is a relative path. The Instruct models' built-in chat template crashes in llama.cpp whenever a request has a system message: it checks `tools is not none`, which is true when `tools` is undefined, then tries to print the missing tools. Every request then fails with **HTTP 400** ("Unable to generate parser for this template… Unknown filter 'tojson' for type Undefined"). `chat_templates/olmo3-instruct-fixed.jinja` is the same template with that check fixed. It produces exactly the prompt Hugging Face's own template does (verified 2026-10-16). Think models don't need it.
```bash
cd ~/"AI Research - Cornell"            # the template path below is relative to this folder
~/src/llama.cpp/build/bin/llama-server \
    -m ~/models/gguf/Olmo-3-7B-Instruct-SFT-Q8_0.gguf \
    --jinja --chat-template-file chat_templates/olmo3-instruct-fixed.jinja -c 8192 --port 8080
# or
~/src/llama.cpp/build/bin/llama-server \
    -m ~/models/gguf/Olmo-3-7B-Instruct-DPO-Q8_0.gguf \
    --jinja --chat-template-file chat_templates/olmo3-instruct-fixed.jinja -c 8192 --port 8080
```
Then send the same `curl` request as Step 8. Instruct models answer directly with no `<think>` block, so expect `reasoning_content` to be empty and the story to be in `content`. Check that `finish_reason` is `stop`.

**Disk budget:** 3 × ~8 GB GGUFs = ~24 GB to keep, plus one 14.6 GB download at a time while converting.

**Step 11: Sample 1,000 stories (Instruct SFT and DPO).** 5 prompts × 100 stories × 2 checkpoints = **1,000 stories**, about 500 words each.

`helper_scripts/generate_stories.py` sends the requests to `llama-server` and saves one JSON line per story. Each line holds:
- the prompt and system prompt
- the checkpoint, revision, and the GGUF the server actually had loaded
- the llama.cpp build
- all sampling settings and the seed
- the raw output, reasoning and story, plus its word count
- `finish_reason`, token counts and speed

Design choices:
- **The five prompts** (`generated_stories/prompts.docx`) are built into the script: "Write a story.", "Write me a story.", "Please write a story.", "Tell me a story.", "Please tell a story."
- **Length:** the prompts don't mention length, so `--target-words 500` adds one line to the system prompt: *"Each story should be about 500 words."* The user prompts stay exactly the paper's. This line is a change from the old run's system prompt; use `--target-words 0` to leave it out.
- **Sampling defaults:** temperature 0.6, top_p 0.95 (the OLMo 3 report's settings), top_k 40, min_p 0.05, max 4,096 tokens. All are flags, so change them once the mentor's script is available.
- **Seeds:** seed = prompt × 100,000 + sample number, so story #17 of "Tell me a story." gets the same seed on both checkpoints.
- **Safety check:** the script stops if the server has a different GGUF loaded than the stage you name.
- **Resumable:** each story is saved the moment it finishes. To stop for the day, press Ctrl-C in both terminals (script and server). To continue, start the server again with the same model, then run the **same generate command with the same `--out` file**. It skips the stories already saved and only redoes the ≤4 that were mid-way when you stopped.
  - A half-written last line from an abrupt stop is dropped automatically.
  - The script refuses to resume if the stage, system prompt or sampling settings differ from what's in the file, so two setups can't get mixed together.
  - Failed requests are skipped and retried on the next run.

*a. Start the server for the first checkpoint* (terminal 1). `--parallel 4` runs four stories at once. The context (`-c 16384`) is shared, so each slot gets 4,096 tokens. The KV cache for 16,384 tokens is about 8 GB on top of the 8 GB model, so quit the Llama desktop app first.
```bash
cd ~/"AI Research - Cornell"            # the template path below is relative to this folder
~/src/llama.cpp/build/bin/llama-server \
    -m ~/models/gguf/Olmo-3-7B-Instruct-SFT-Q8_0.gguf \
    --jinja --chat-template-file chat_templates/olmo3-instruct-fixed.jinja \
    -c 16384 --parallel 4 --port 8080 2>&1 | tee ~/models/server-instruct-sft.log
```

*b. Generate 500 stories* (terminal 2, once `curl http://localhost:8080/health` says ok):
```bash
cd ~/"AI Research - Cornell"
uv run --no-project helper_scripts/generate_stories.py instruct-sft \
    --out new_stories/data/instruct-sft.jsonl --n-per-prompt 100 --target-words 500 --workers 4
```
Try a few first: `--n-per-prompt 2` writes 10 stories into the same file, then rerun with `100` to finish the rest.

*c. Switch to DPO:* Ctrl-C the server and restart it with the same command, swapping in `Olmo-3-7B-Instruct-DPO-Q8_0.gguf` (keep `--chat-template-file`; the DPO template has the same bug). Then:
```bash
uv run --no-project helper_scripts/generate_stories.py instruct-dpo \
    --out new_stories/data/instruct-dpo.jsonl --n-per-prompt 100 --target-words 500 --workers 4
```

*d. Check the results:*
```bash
# count, finish reasons, and average length per file
python3 -c "
import json, sys, collections
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    words = [r['story_words'] for r in rows]
    print(f, len(rows), 'stories', dict(collections.Counter(r['finish_reason'] for r in rows)),
          'mean words', round(sum(words) / len(words)))
" new_stories/data/instruct-*.jsonl

# Core rates (the story text is in field "story")
uv run --no-project helper_scripts/core_rates.py jsonl new_stories/data --field story
```

**Time:** the RLVR smoke test ran at about 11 tokens/s for one request. 500 words is about 700 tokens, so that's roughly 1 minute per story, or **~8 hours per checkpoint** one at a time. `--parallel 4` should cut this a lot; the script prints a running "min left" estimate. Keep the Mac awake with `caffeinate -i` in front of the generate command.

`new_stories/data/` is ignored by git (the `data/` rule). Add `new_stories/data/` to Obsidian's excluded files too.

### Day 4–5: Get to know the data

1. **Read about 50 of the existing OLMo stories** (`generated_stories/data/…Olmo-3-7B-Think…jsonl`, field `content`), plus about 20 from each of the other models. Take notes:
   - What does a typical "Core story" look like? What else recurs that *isn't* in the Core list?
   - Where does the scorer go wrong? It uses whole-word matching, so "goalkeeper" doesn't count, but "Mayor" as a surname or "conductor" in physics would.
2. **Skim the Think training extracts** (`dolci-think-sft-7b.jsonl`, `dolci-think-dpo-7b.jsonl`; field `text`). Run the scorer on them (`--field text`) and compare with Table 2. Read about 20 records, keeping in mind that some aren't stories and some are explicit.
3. **Look at `story_metadata.csv`**: the model-extracted names, settings and professions for every story. Join on model label + `story_index` ↔ `i`.

### Questions to bring to your mentor

1. Can I see **the saved OLMo generation script**? I want the exact sampling settings (temperature, top_p, top_k, min_p, repeat penalty, seed handling) and the llama.cpp version, so the new runs match.
2. Is there a **GPU machine or cluster allocation** I should use, or is laptop-scale fine for a pilot?
3. How many stories per checkpoint? (Suggestion: 5 prompts × 200 = 1,000 per checkpoint, which gives roughly ±1.5 percentage points of error on Core %.)
4. Should the **base model** (before SFT) or the **Instruct** branch be included, or only the 3 Think checkpoints for now?
5. Should I keep the **4,096-token cap**? Some checkpoints may not finish their reasoning in time. Should truncated outputs be dropped, counted separately, or regenerated with a higher cap?

---

## 5. End-of-week checklist

- [ ] Read the paper, both data READMEs and the OLMo 3 post-training sections. Wrote notes on each training stage.
- [ ] Ran `core_rates.py` and matched Table 5's OLMo column.
- [ ] llama.cpp built; commit hash recorded.
- [ ] All three checkpoints downloaded, converted to Q8_0 GGUF, and smoke-tested with the system prompt.
- [ ] Speed measured; time estimate for the full run written down.
- [ ] Read about 50 OLMo stories and about 20 training records; notes on patterns and scorer errors.
- [ ] Met with mentor; questions above answered.

**Next (Phase 1, to be written):** a generation script that saves everything, a 750-story pilot (50 per prompt per checkpoint), then the full run.
