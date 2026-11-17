# AI Research - Cornell: When do OLMo 3 models start writing "Core" stories?

Mentors / paper: Sil Hamilton & David Mimno, *Elias in the Lighthouse, Again? Diagnosing Low Diversity in LLM Stories* (2026).
Full plan: `PROJECT_SCOPE.md`. Session history: `CC-Session-Logs/` (CPR plugin).

## Research question (PROTECTED)
Give the same prompts with the same settings to three OLMo 3 7B Think checkpoints (SFT, DPO, RLVR/final) and find the stage where "Core" stories (Elias / Mara / Elara, keeper / baker / mayor / clockmaker / fisherman / librarian / conductor, lighthouse) appear or become much more common.

- Core share of the *training data* rises only modestly: SFT 3.0% → DPO 8.0% → RL 10.9% (paper Table 2).
- The final model writes Core stories 73.1% of the time (existing 5,000-story OLMo run).
- A jump in outputs much larger than in the data points to the training method amplifying the pattern. This sets up Question 3: do Core stories get extra weight during DPO?

## Rules (PROTECTED)
- **Score only the story.** Never score the prompt or the `<think>` reasoning.
- **Same setup for every checkpoint:** llama.cpp, Q8_0 GGUF, same system prompt, same sampling settings, same token cap.
- **Use all 5 prompts** (`generated_stories/prompts.docx`) for every checkpoint, not the single "write a story" prompt from the old run.
- **New runs must save:** prompt, checkpoint + revision, all sampling settings, raw reasoning, final story, finish reason.
- **Keep models outside this folder** (`~/models`, GGUFs in `~/models/gguf`). Never put models or large generated outputs in the iCloud Obsidian vault.
- Study the **Think** branch only. Ignore `dolci-instruct-*` files for now.
- Some training extracts are explicit or aren't stories at all.

## Checkpoints
Pinned revisions are in `helper_scripts/download_models.py`. Each download is about 14.6 GB; each Q8_0 GGUF about 8 GB.

| Stage | Hugging Face repo |
|---|---|
| sft | `allenai/Olmo-3-7B-Think-SFT` |
| dpo | `allenai/Olmo-3-7B-Think-DPO` |
| rlvr | `allenai/Olmo-3-7B-Think` |

## Generation setup
System prompt (use exactly):
```
You are a fiction writer.
When the user asks for a story, respond with only the story text. Do not ask clarifying questions. Do not add introductions, markdown separators, bullet lists, or follow-up suggestions.
```
- Old run: max output 4,096 tokens, everything up to `</think>` removed before saving.
- OLMo 3 report evaluates at temperature 0.6, top_p 0.95, 32K max tokens. Final settings are pending the mentor's script.
- Machine: Apple M4, 24 GB memory, about 144 GB free disk. `uv`, `git`, Homebrew installed.
- llama.cpp: built at commit `5cdadfc32` (2026-10-14, build b11633) in `~/src/llama.cpp`.
- **Instruct GGUFs need `--chat-template-file chat_templates/olmo3-instruct-fixed.jinja`** (run llama-server from this folder). The built-in template checks `tools is not none`, which fails when llama.cpp leaves `tools` undefined, so every request with a system message returns HTTP 400. The fixed template renders identically to Hugging Face's (verified 2026-10-16).
- All three GGUFs (Think, Instruct-SFT, Instruct-DPO) were verified against the source weights (corr 0.99998) and pass the "Paris" check. The safetensors downloads were deleted 2026-10-16; re-download with `download_models.py` if needed.

## Conversion gotcha (PROTECTED)
- **The conversion venv (`~/src/llama.cpp/.venv`) must use Python 3.12, not Homebrew's 3.14.** llama.cpp pins `numpy~=2.2.6`, which silently miscomputes large arrays on Python 3.14. The first RLVR Q8_0 GGUF had ~50% of weight blocks sign-flipped and the model output only "@@@@". Rebuilt with `uv venv --python 3.12 .venv` (2026-10-14).
- After every conversion, sanity-check before trusting it: `./build/bin/llama-completion -m <gguf> -n 12 --temp 0 -p "The capital of France is" -no-cnv` should continue with "Paris".

## Folder layout
- `helper_scripts/core_rates.py`: counts the 11 Core words (whole-word matching).
- `helper_scripts/download_models.py`: downloads the pinned checkpoints.
- `generated_stories/data/`: 4 × 5,000 stories (Claude, Gemini, GPT, OLMo) + `story_metadata.csv`. Story text is in field `content`.
- `training_stories/data/`: Dolci SFT / DPO / RL extracts. Text is in field `text`.
- `Notes/`: reading notes (Obsidian). OLMo 3 Think report notes start at `Notes/OLMo 3 Think - Overview.md`.
- The top-level folder name has spaces (`~/"AI Research - Cornell"`), so quote it in shell commands.

Verify the scorer (reproduces Table 5's OLMo column; verified 2026-10-14). Output goes to `core_rates.json`; add `--field text` for the training extracts:
```bash
uv run --no-project helper_scripts/core_rates.py jsonl generated_stories/data > core_rates.json
```

## Current phase: Week 1 (Phase 0, background and setup)

### Day 1-2: Reading
- [ ] OLMo 3 tech report, Think post-training (§4). Claude's summary notes are in `Notes/OLMo 3 Think - *.md`.
- [ ] The paper: §3-4, Table 2, Table 5
- [ ] The two data READMEs (`generated_stories/README.pdf`, `training_stories/README.pdf`)
- [ ] Skim InstructGPT (Ouyang 2022), DPO (Rafailov 2023, intro + main equation), Kirk et al. 2024
- [ ] llama.cpp README: building, `convert_hf_to_gguf.py`, `llama-server`

### Day 2-3: Environment setup
- [ ] `git init` with `.gitignore` for `data/`, models, `*.gguf`, `*.safetensors`, `.obsidian/workspace*.json`
- [x] Run `core_rates.py` and match Table 5. Passed 2026-10-14: OLMo 73.1% Core; Elara 6,625.5 PPM, Elias 893.2, lighthouse 1,958.7, keeper 713.5 (exact match). Output in `core_rates.json`.
- [x] Build llama.cpp from source in `~/src/llama.cpp`; commit `5cdadfc32` (first build failed on a stale `/Library/Developer/CommandLineTools/usr/include/c++` dir that hid the C++ headers)
- [ ] Download `rlvr`, convert to Q8_0 GGUF, smoke test with `llama-server --jinja -c 8192 --port 8080`
- [ ] Check: reasoning inline vs `reasoning_content`; `finish_reason` is `stop`; speed and server sampling defaults
- [ ] Time estimate from 5-10 stories (tokens/s, tokens per response, hours for 1,000 stories × 3 checkpoints); try `--parallel N`
- [ ] Repeat for `sft` and `dpo`. Does SFT close its `<think>` block and write a story?

### Day 4-5: Get to know the data
- [ ] Read about 50 OLMo stories and about 20 from each other model; note recurring patterns and scorer errors
- [ ] Run the scorer on Think SFT/DPO extracts (`--field text`), compare with Table 2, read about 20 records
- [ ] Explore `story_metadata.csv` (join model label + `story_index` ↔ `i`)

### Questions for the mentor
1. Can I see the saved OLMo generation script (sampling settings, seed handling, llama.cpp version)?
2. Is there a GPU machine or cluster allocation, or is laptop scale fine for a pilot?
3. How many stories per checkpoint? (Suggested: 5 prompts × 200 = 1,000.)
4. Include the base model or the Instruct branch, or only the 3 Think checkpoints?
5. Keep the 4,096-token cap? How to handle truncated outputs?
6. Should we score Qwen3-32B vs Qwen3-0.6B stories, and the chosen vs rejected sides of `dolci-think-dpo-7b.jsonl` separately? (DPO pairs Qwen3-32B chosen vs Qwen3-0.6B rejected.)

**Next (Phase 1):** a generation script that saves everything, a 750-story pilot (50 per prompt per checkpoint), then the full run.

## Workflow
- Start Claude Code from this folder so CPR logs land in `CC-Session-Logs/`.
- This folder is also an Obsidian vault. Exclude `training_stories/` and `generated_stories/data/` in Obsidian settings.
- End sessions with `/cpr:compress`; record lasting decisions with `/cpr:preserve`.
