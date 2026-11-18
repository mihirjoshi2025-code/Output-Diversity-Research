# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "huggingface_hub"]
# ///
"""
Sample first-turn (prompt, response) story pairs from WildChat-4.8M-Full for a safety-classifier check.

Streams --files parquet shards from the pinned revision (each ~110 MB is read into memory and dropped; nothing big
is written), keeps English conversations whose first user message is a story request (same check as notebook 03),
and writes:
  data/wildchat/response_pool.parquet   every candidate pair, with WildChat's own flags and moderation scores
  data/wildchat/response_sample.jsonl   a seeded, balanced sample: --per-group pairs WildChat calls safe
                                        (whole conversation not toxic) and --per-group whose response is toxic

WildChat flags a message as toxic when OpenAI moderation flags it OR Detoxify toxicity > 0.1 (this rule reproduces
the flag on 100% of shard 0's 236K messages, checked 2026-11-17); a conversation is toxic if any message is. The
moderation output includes `illicit`, so it is OpenAI's omni-moderation model (2024+), not the 2023 endpoint.
`response_toxic` is the flag on the assistant reply. Raw scores are kept as oai_<category> (OpenAI category
scores, 0-1) and detox_<field> (Detoxify scores, 0-1).

    uv run helper_scripts/wildchat_response_sample.py --per-group 50
"""

import argparse
import io
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "wildchat"
WILDCHAT = ("yuntian-deng/WildChat-4.8M-Full", "462d2dc4d4189436bc0d54313b4fd925f1fe93eb")
OAI_CATS = ["harassment", "harassment_threatening", "hate", "hate_threatening", "illicit", "illicit_violent",
            "self_harm", "self_harm_instructions", "self_harm_intent", "sexual", "sexual_minors", "violence",
            "violence_graphic"]
DETOX = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]
COLS = ["conversation_hash", "model", "timestamp", "language", "toxic", "conversation", "openai_moderation",
        "detoxify_moderation"]

# Same story-request check as notebooks/03_wildchat_toxicity_provenance.ipynb (section 5).
STORY_WORDS = re.compile(
    r"\b(?:stor(?:y|ies)|tale|fable|fanfic\w*|fan[- ]fiction|narrative|novel|chapter|plot|script|screenplay|"
    r"episode|scene|scenario|dialog(?:ue)?|backstory|crossover|roleplay|role-play|fiction|bedtime|legend|myth|saga)\b",
    re.I)
SCRIPT_FORMAT = re.compile(r"^\s*[(\[]|\b[A-Z][\w.\-]*(?: [A-Z][\w.\-]*)?: [\"“]")


def is_story_request(prompt: str) -> bool:
    return bool(STORY_WORDS.search(prompt) or SCRIPT_FORMAT.search(prompt))


def first_pair(row) -> dict | None:
    conv = row["conversation"]
    if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
        return None
    prompt, response = conv[0]["content"] or "", conv[1]["content"] or ""
    if not prompt.strip() or not response.strip():
        return None
    mods, detox = row["openai_moderation"], row["detoxify_moderation"]
    rmod = mods[1] if len(mods) > 1 else None
    rdet = detox[1] if len(detox) > 1 else None
    scores = (rmod or {}).get("category_scores") or {}
    raw = {f"oai_{c}": scores.get(c) for c in OAI_CATS} | {f"detox_{f}": (rdet or {}).get(f) for f in DETOX}
    return raw | {
        "key": f"wc:{row['conversation_hash']}",
        "conversation_hash": row["conversation_hash"],
        "gpt_model": row["model"],
        "timestamp": str(row["timestamp"]),
        "conversation_toxic": bool(row["toxic"]),
        "prompt_toxic": bool(conv[0]["toxic"]),
        "response_toxic": bool(conv[1]["toxic"]),
        "response_openai_flagged": bool(rmod["flagged"]) if rmod else None,
        "response_openai_categories": [k for k, v in rmod["categories"].items() if v and "/" not in k] if rmod else [],
        "response_detoxify_toxicity": float(rdet["toxicity"]) if rdet else None,
        "prompt": prompt,
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", type=int, default=1, help="How many parquet shards to read (37K rows each)")
    parser.add_argument("--per-group", type=int, default=50, help="Pairs per group in the sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fs = HfFileSystem()
    repo, rev = WILDCHAT
    shards = sorted(fs.glob(f"datasets/{repo}@{rev}/data/*.parquet"))[: args.files]
    rows = []
    for path in shards:
        print(f"reading {Path(path).name}", flush=True)
        df = pq.read_table(io.BytesIO(fs.cat_file(path)), columns=COLS).to_pandas()
        df = df[df.language == "English"]
        rows += [p for p in (first_pair(r) for _, r in df.iterrows()) if p and is_story_request(p["prompt"])]
    pool = pd.DataFrame(rows).drop_duplicates("conversation_hash")
    pool["shard_revision"] = rev

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT_DIR / "response_pool.parquet")
    safe = pool[~pool.conversation_toxic]
    toxic = pool[pool.response_toxic]
    print(f"{len(pool):,} English first-turn story pairs: {len(safe):,} WildChat-safe, "
          f"{len(toxic):,} with a toxic-flagged response")

    sample = pd.concat([
        safe.sample(args.per_group, random_state=args.seed).assign(group="wildchat_safe"),
        toxic.sample(args.per_group, random_state=args.seed).assign(group="wildchat_toxic_response"),
    ])
    out = OUT_DIR / "response_sample.jsonl"
    sample.to_json(out, orient="records", lines=True, force_ascii=False)
    print(f"{len(sample)} pairs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
