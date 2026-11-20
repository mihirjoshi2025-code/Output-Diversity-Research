"""
Run TopicGPT (Pham et al., NAACL 2024; github.com/chtmp223/topicGPT) on the WildChat story-keyword prompts, with a
local model, so its bottom-up topics can be compared with the fixed 10-topic list Jev assigned (jev_classify.py
--task topic).

Backend: TopicGPT's "ollama" API, which is just an OpenAI-compatible client at localhost:11434/v1. llama-server
serves the same API, so it stands in for Ollama (no OpenAI key, nothing leaves the machine):

    ~/src/llama.cpp/build/bin/llama-server -m ~/models/gguf/Olmo-3-7B-Instruct-DPO-Q8_0.gguf --jinja \
        --chat-template-file chat_templates/olmo3-instruct-fixed.jinja -c 32768 --parallel 4 --port 11434

Environment: TopicGPT hard-requires vLLM (GPU only) and imports Vertex/Gemini/Anthropic clients at load time, so it
is installed without dependencies in its own venv, and those cloud-only modules are stubbed below (the ollama path
never touches them). Set up once:

    git clone https://github.com/chtmp223/topicGPT ~/src/topicGPT        # prompts live in the repo, not the wheel
    cd ~/src/topicGPT && uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python --no-deps topicgpt_python==0.2.7
    uv pip install --python .venv/bin/python anytree "numpy<2" "openai>=1.54,<2" "pandas<3" regex \
        "tiktoken>=0.7,<0.8" tqdm "sentence-transformers<4" pyyaml tenacity scikit-learn pyarrow

Changes from TopicGPT's defaults (everything else, including its prompts' wording and parsing, is unchanged):
  - documents are the prompts cut to their first 1,500 + last 700 characters (TopicGPT assumes a 128K context for
    non-GPT models and would not cut them; the request is usually at the start or the end);
  - the generation prompt gets one added paragraph saying the documents are chatbot requests and topics should
    name the subject matter, not the task ("Creative writing" would otherwise swallow everything);
  - the seed topic is "Sports" instead of "Trade": same granularity hint, and it isn't one of Jev's 10 topics,
    so the taxonomy TopicGPT finds is independent of ours.

Stages (each resumes from its output files):
    PY=~/src/topicGPT/.venv/bin/python
    $PY helper_scripts/topicgpt_run.py prep        # docs + generation sample + prompts
    $PY helper_scripts/topicgpt_run.py generate    # level-1 topics from a 500-prompt seeded sample
    $PY helper_scripts/topicgpt_run.py refine      # merge near-duplicate topics, drop ones under 1% of mentions
    $PY helper_scripts/topicgpt_run.py assign      # assign every prompt (16 chunks, 4 at a time)
    $PY helper_scripts/topicgpt_run.py correct     # re-ask prompts whose assignment isn't in the topic list
"""

import argparse
import json
import random
import shutil
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "data" / "wildchat"
OUT = WC / "topicgpt"
PROMPTS = OUT / "prompts"
REPO_PROMPTS = Path.home() / "src" / "topicGPT" / "prompt"
MODEL = "olmo-3-7b-instruct-dpo-q8_0"  # sent to llama-server, which ignores it; recorded for provenance
HEAD, TAIL, GEN_SAMPLE, CHUNKS, WORKERS, SEED = 1_500, 700, 500, 16, 4, 0

GEN_NOTE = (
    "The documents are requests that users sent to an AI chatbot; most ask for a story, roleplay or script. "
    "Topics must name the SUBJECT MATTER or genre of the requested content (what it is about), not the task type: "
    "do not use labels such as \"Storytelling\", \"Creative Writing\", \"Fiction\" or \"Roleplay\". A request "
    "that is not asking for fiction gets a topic for what it does ask for.\n\n"
)
SEED_TOPIC = "[1] Sports: Mentions athletes, games, matches, or competitions.\n"


def stub_cloud_modules() -> None:
    """Let topicgpt_python import without vLLM / Vertex / Gemini / Anthropic installed."""
    names = {
        "vllm": ["LLM", "SamplingParams"],
        "vertexai": ["init"],
        "vertexai.generative_models": ["GenerationConfig", "GenerativeModel", "HarmBlockThreshold", "HarmCategory",
                                       "SafetySetting"],
        "anthropic": ["AnthropicVertex"],
        "google": [],
        "google.generativeai": ["GenerativeModel", "configure"],
        "google.generativeai.types": ["HarmCategory", "HarmBlockThreshold"],
    }
    for name, attrs in names.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for a in attrs:
            setattr(mod, a, type(a, (), {}))
        sys.modules[name] = mod
    sys.modules["vertexai"].generative_models = sys.modules["vertexai.generative_models"]
    sys.modules["google"].generativeai = sys.modules["google.generativeai"]
    sys.modules["google.generativeai"].types = sys.modules["google.generativeai.types"]


def cut(text: str) -> str:
    return text if len(text) <= HEAD + TAIL else text[:HEAD] + "\n[...]\n" + text[-TAIL:]


def prep() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROMPTS.mkdir(exist_ok=True)
    docs = [{"id": r["key"], "text": cut(r["prompt"])} for r in map(json.loads, (WC / "response_pool.jsonl").open())]
    with (OUT / "docs.jsonl").open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    sample = random.Random(SEED).sample(docs, GEN_SAMPLE)
    with (OUT / "generation_sample.jsonl").open("w") as f:
        for d in sample:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    for name in ["generation_1.txt", "refinement.txt", "assignment.txt", "correction.txt"]:
        shutil.copy(REPO_PROMPTS / name, PROMPTS / name)
    gen = (PROMPTS / "generation_1.txt").read_text()
    first_break = gen.index("\n\n") + 2
    (PROMPTS / "generation_1.txt").write_text(gen[:first_break] + GEN_NOTE + gen[first_break:])
    (PROMPTS / "seed_1.md").write_text(SEED_TOPIC)
    print(f"{len(docs):,} docs, {len(sample)} in the generation sample -> {OUT.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prep", "generate", "refine", "assign", "correct"])
    args = parser.parse_args()
    if args.stage == "prep":
        return prep()

    stub_cloud_modules()
    import pandas as pd
    from topicgpt_python import assign_topics, correct_topics, generate_topic_lvl1, refine_topics

    if args.stage == "generate":
        generate_topic_lvl1("ollama", MODEL, str(OUT / "generation_sample.jsonl"), str(PROMPTS / "generation_1.txt"),
                            str(PROMPTS / "seed_1.md"), str(OUT / "generation_1.jsonl"), str(OUT / "topics_1.md"),
                            verbose=False)
    elif args.stage == "refine":
        refine_topics("ollama", MODEL, str(PROMPTS / "refinement.txt"), str(OUT / "generation_1.jsonl"),
                      str(OUT / "topics_1.md"), str(OUT / "refined_topics_1.md"), str(OUT / "refined_generation_1.jsonl"),
                      verbose=False, remove=True, mapping_file=str(OUT / "refinement_mapping.json"))
    elif args.stage == "assign":
        docs = pd.read_json(OUT / "docs.jsonl", lines=True)
        parts = OUT / "assignment_parts"
        parts.mkdir(exist_ok=True)
        jobs = []
        for i in range(CHUNKS):
            chunk_in, chunk_out = parts / f"docs_{i}.jsonl", parts / f"assignment_{i}.jsonl"
            if chunk_out.exists():
                continue
            docs.iloc[i::CHUNKS].to_json(chunk_in, orient="records", lines=True, force_ascii=False)
            jobs.append((chunk_in, chunk_out))
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(lambda io: assign_topics("ollama", MODEL, str(io[0]), str(PROMPTS / "assignment.txt"),
                                                   str(io[1]), str(OUT / "refined_topics_1.md"), verbose=False), jobs))
        merged = pd.concat(pd.read_json(parts / f"assignment_{i}.jsonl", lines=True) for i in range(CHUNKS))
        merged.to_json(OUT / "assignment.jsonl", orient="records", lines=True, force_ascii=False)
        print(f"{len(merged):,} assignments -> {(OUT / 'assignment.jsonl').relative_to(ROOT)}")
    elif args.stage == "correct":
        correct_topics("ollama", MODEL, str(OUT / "assignment.jsonl"), str(PROMPTS / "correction.txt"),
                       str(OUT / "refined_topics_1.md"), str(OUT / "assignment_corrected.jsonl"), verbose=False)


if __name__ == "__main__":
    main()
