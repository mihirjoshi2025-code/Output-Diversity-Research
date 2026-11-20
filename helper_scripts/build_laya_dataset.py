# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow"]
# ///
"""
Export the one WildChat shard we labelled (train-00000 at the pinned revision; see wildchat_response_sample.py) with
all of Jev's labels, as a self-contained dataset to share (e.g. for fine-tuning Laya on Jev's labels).

One row per English first-turn story-keyword (prompt, response) pair: the raw text, the exact text Jev was shown,
Jev's labels and probabilities, and WildChat's own flags and moderation scores. No train/test split: whoever
fine-tunes makes their own.

Writes to --out (default data/wildchat/jev_shard0/):
    wildchat_shard0_jev_labels.jsonl     one JSON object per line (pandas, Hugging Face `datasets`, any language)
    wildchat_shard0_jev_labels.parquet   the same rows, smaller and typed
    README.md                            what every field means, how the rows were chosen, caveats

    uv run helper_scripts/build_laya_dataset.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from safety_common import PROMPT_CHARS, RESPONSE_CHARS, truncate_pair  # noqa: E402
from jev_classify import CATEGORIES, SAFETY_CRITERIA, TOPICS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "data" / "wildchat"
NAME = "wildchat_shard0_jev_labels"


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.open()])


def build() -> pd.DataFrame:
    pool = pd.read_parquet(WC / "response_pool.parquet")
    safety = read_jsonl(WC / "jev_pool_labels.jsonl")
    topics = read_jsonl(WC / "jev_topics.jsonl")
    assert len(pool) == len(safety) == len(topics) == pool.key.nunique(), "label files don't cover the pool 1:1"
    assert (safety.model == "jev-1.14.0").all() and (topics.model == "jev-1.14.0").all()

    safety = safety.rename(columns={"truncated": "text_truncated", "model": "jev_model"})[
        ["key", "safety", "safety_confidence", "safety_probabilities", "category", "category_probabilities",
         "refusal", "refusal_probabilities", "text_truncated", "jev_model"]]
    topics = topics[["key", "topic", "topic_confidence", "topic_probabilities", "fan_fiction", "jailbreak"]]
    df = pool.merge(safety, on="key", validate="1:1").merge(topics, on="key", validate="1:1")

    cut = [truncate_pair(p, r) for p, r in zip(df.prompt, df.response)]
    df["jev_input_safety"] = [f"USER: {p}\n\nASSISTANT: {r}" for p, r, _ in cut]
    df["jev_input_topic"] = [f"USER: {truncate_pair(p, '')[0]}" for p in df.prompt]
    df["is_fiction"] = df.topic != "Not a fiction request"
    df["wildchat_shard"] = "data/train-00000"
    df = df.rename(columns={"shard_revision": "wildchat_revision", "gpt_model": "wildchat_gpt_model",
                            "timestamp": "wildchat_timestamp"})

    first = ["key", "conversation_hash", "prompt", "response", "jev_input_safety", "jev_input_topic",
             "safety", "safety_confidence", "category", "refusal", "topic", "topic_confidence", "is_fiction",
             "fan_fiction", "jailbreak", "safety_probabilities", "category_probabilities", "refusal_probabilities",
             "topic_probabilities", "text_truncated", "jev_model",
             "conversation_toxic", "prompt_toxic", "response_toxic", "response_openai_flagged",
             "response_openai_categories", "response_detoxify_toxicity"]
    rest = sorted(c for c in df.columns if c.startswith(("oai_", "detox_")))
    tail = ["wildchat_gpt_model", "wildchat_timestamp", "wildchat_shard", "wildchat_revision"]
    return df[first + rest + tail].sort_values("key").reset_index(drop=True)


def readme(df: pd.DataFrame) -> str:
    counts = lambda s: "\n".join(f"| {k} | {v:,} |" for k, v in s.value_counts().items())  # noqa: E731
    topic_lines = "\n".join(f"- **{k}**: {v}" for k, v in TOPICS.items())
    safety_lines = "\n".join(f"- **{k}**: {v}" for k, v in SAFETY_CRITERIA.items())
    return f"""# WildChat shard 0: story pairs labelled by Jev

{len(df):,} (prompt, response) pairs from one shard of
[WildChat-4.8M-Full](https://huggingface.co/datasets/yuntian-deng/WildChat-4.8M-Full)
(`{df.wildchat_shard[0]}`, revision `{df.wildchat_revision[0]}`), each labelled by TypeSafe's Jev
(`{df.jev_model[0]}`) for reply safety and prompt topic. Built by `helper_scripts/build_laya_dataset.py`.

Content warning: many rows are explicit sexual, fetish, or violent text. Keep the sharing link restricted.

## Files

| File | Use |
|---|---|
| `{NAME}.jsonl` | One JSON object per line. `pd.read_json(path, lines=True)` or `datasets.load_dataset("json", data_files=path)`. |
| `{NAME}.parquet` | Same rows, smaller. `pd.read_parquet(path)`. |

Both hold identical data. There is no train/test split.

## How rows were chosen

From the shard's ~37K conversations: English, the first message is from the user and the second from the assistant,
both non-empty, and the user's first message matches a story-keyword regex (story, tale, fanfic, script, scene,
roleplay, ...). Only that first exchange is kept. The keyword filter is loose: Jev marks **{(~df.is_fiction).sum():,}
({(~df.is_fiction).mean():.0%})** of rows "Not a fiction request" (code, essays, agent prompts). Use `is_fiction` to
keep stories only.

## Inputs and labels

**Inputs.** `prompt` and `response` are the raw text. `jev_input_safety` / `jev_input_topic` are the exact strings
Jev was sent: prompts over {PROMPT_CHARS:,} characters keep their first and last halves around `[...]`, responses
are cut at {RESPONSE_CHARS:,} characters (`text_truncated` marks the {df.text_truncated.sum()} rows where either
was cut). Train on these to see what Jev saw.

**Reply safety** (from `jev_input_safety`; Qwen3Guard-Gen's response-moderation policy, asked of Jev verbatim):
{safety_lines}

| `safety` | rows |
|---|---|
{counts(df.safety)}

- `category`: the single unsafe category that best fits the reply, or `None`. Choices: {", ".join(CATEGORIES)}, None.
- `refusal`: `Yes` if the reply refuses the request.
- `*_probabilities`: Jev's full distribution over the choices (dict), usable as soft labels. `*_confidence` is
  Jev's confidence in its top choice.

**Prompt topic** (from `jev_input_topic`, prompt only, so it doesn't depend on the reply):
{topic_lines}

| `topic` | rows |
|---|---|
{counts(df.topic)}

- `fan_fiction`, `jailbreak`: Jev's probability (0-1) that the prompt uses an existing franchise / tries to get
  around the AI's rules.

## WildChat's own labels (for comparison, not targets)

- `response_toxic`: WildChat's flag on the reply = OpenAI omni-moderation flagged it OR Detoxify toxicity > 0.1
  (this rule reproduces the flag on 100% of the shard's messages). `prompt_toxic`, `conversation_toxic` likewise.
- `response_openai_flagged`, `response_openai_categories`: OpenAI moderation's verdict and flagged categories.
- `oai_<category>`: OpenAI moderation scores (0-1). `detox_<field>`: Detoxify scores (0-1).
- `wildchat_gpt_model`, `wildchat_timestamp`: which GPT model replied, and when.

## Caveats

- Jev's labels are a model's labels, not ground truth: Jev agrees with Qwen3Guard-Gen-4B on safe-vs-not 92% of the
  time (kappa 0.83, 100 pairs) but has not yet been checked against human labels. A model trained on these labels
  learns Jev's mistakes too.
- Rare classes are small: `Controversial` {(df.safety == "Controversial").sum()}, topic `Other`
  {(df.topic == "Other").sum()}, `Romance (non-explicit)` {(df.topic == "Romance (non-explicit)").sum()}.
- One shard of 127, so not necessarily representative of all of WildChat.
- Other shards must go through the same filter before a model trained here labels them.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=WC / "jev_shard0")
    args = parser.parse_args()
    df = build()
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_json(args.out / f"{NAME}.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_parquet(args.out / f"{NAME}.parquet", index=False)
    (args.out / "README.md").write_text(readme(df))
    print(f"{len(df):,} rows x {df.shape[1]} columns -> {args.out}")


if __name__ == "__main__":
    main()
