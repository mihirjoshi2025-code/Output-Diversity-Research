# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Sample stories from a running llama-server and save everything to JSONL.

Each line records the prompt, checkpoint + revision, all sampling settings, the seed,
the raw output, the reasoning, the story, the finish reason, token counts and timings.
Re-running with the same --out skips stories that are already saved, so a stopped run
can be resumed.
"""

import argparse
import json
import re
import sys
import threading
import time
import hashlib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_models import MODELS  # noqa: E402

# generated_stories/prompts.docx (the paper's five prompts)
PROMPTS = [
    "Write a story.",
    "Write me a story.",
    "Please write a story.",
    "Tell me a story.",
    "Please tell a story.",
]
SYSTEM_PROMPT = (
    "You are a fiction writer.\n"
    "When the user asks for a story, respond with only the story text. Do not ask clarifying "
    "questions. Do not add introductions, markdown separators, bullet lists, or follow-up suggestions."
)


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:  # surface the server's own explanation, not just "400 Bad Request"
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail[:500]}") from None


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def split_reasoning(message: dict) -> tuple[str, str]:
    """Return (reasoning, story). Handles reasoning in its own field or inline in <think>."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if "</think>" in content:
        head, _, content = content.partition("</think>")
        reasoning = (reasoning + "\n" + head.replace("<think>", "")).strip()
    return reasoning, content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=tuple(MODELS), help="Checkpoint the server is running (recorded, not loaded)")
    parser.add_argument("--out", type=Path, required=True, help="JSONL output file (appended to)")
    parser.add_argument("--n-per-prompt", type=int, default=100)
    parser.add_argument("--target-words", type=int, default=500,
                        help="Adds 'Each story should be about N words.' to the system prompt; 0 = no length line")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent requests; match llama-server --parallel")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    repo, revision = MODELS[args.stage]
    system = SYSTEM_PROMPT
    if args.target_words:
        system += f"\nEach story should be about {args.target_words} words."
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                "min_p": args.min_p, "max_tokens": args.max_tokens}

    # What the server actually has loaded, so the record can't silently point at the wrong model.
    props = get(f"{args.url}/props")
    server_model = props.get("model_path") or get(f"{args.url}/v1/models")["data"][0]["id"]
    server_build = props.get("build_info")
    template = props.get("chat_template") or ""
    template_sha = hashlib.sha256(template.encode()).hexdigest()[:12] if template else None
    expected = repo.split("/")[1]
    if not Path(server_model).name.lower().startswith(expected.lower() + "-q8_0"):
        sys.exit(f"Server has {server_model!r} loaded, but stage {args.stage!r} expects {expected}. "
                 "Restart llama-server with the right GGUF.")

    done = set()
    if args.out.exists():
        lines = args.out.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):  # stopped mid-write: drop the partial line
            print("  dropping a half-written last line from the previous run", flush=True)
            lines.pop()
            args.out.write_text("".join(lines))
        for line in lines:
            rec = json.loads(line)
            # Resuming with different settings would silently mix two experiments in one file.
            for key, now in (("stage", args.stage), ("system_prompt", system), ("sampling", sampling)):
                if rec[key] != now:
                    sys.exit(f"{args.out} was made with a different {key}:\n  file: {rec[key]!r}\n  now:  {now!r}\n"
                             "Use the original flags, or a new --out file.")
            done.add((rec["prompt_index"], rec["sample_index"]))
    jobs = [(p, s) for p in range(len(PROMPTS)) for s in range(args.n_per_prompt) if (p, s) not in done]
    print(f"{args.stage}: {len(done)} already saved, {len(jobs)} to go -> {args.out}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()

    def run(job: tuple[int, int]) -> dict:
        p, s = job
        seed = p * 100_000 + s  # same seed for the same (prompt, sample) on every checkpoint
        body = {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": PROMPTS[p]}],
                "seed": seed, **sampling}
        resp = post(f"{args.url}/v1/chat/completions", body, args.timeout)
        choice = resp["choices"][0]
        reasoning, story = split_reasoning(choice["message"])
        return {
            "stage": args.stage, "repo": repo, "revision": revision,
            "server_model": server_model, "llama_cpp_build": server_build,
            "chat_template_sha256": template_sha,
            "prompt_index": p, "sample_index": s, "prompt": PROMPTS[p], "system_prompt": system,
            "seed": seed, "sampling": sampling,
            "raw_content": choice["message"].get("content"),
            "reasoning": reasoning, "story": story, "story_words": len(story.split()),
            "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage"), "timings": resp.get("timings"),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    start = time.time()
    errors = 0
    with args.out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:  # keep going; failed ones are retried on the next run
                print(f"  error: {e}", file=sys.stderr, flush=True)
                errors += 1
                if errors >= 10 and errors == n:
                    sys.exit("First 10 requests all failed; stopping. Fix the error above and rerun.")
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            rate = (time.time() - start) / n
            print(f"  [{n}/{len(jobs)}] prompt {rec['prompt_index']} #{rec['sample_index']}: "
                  f"{rec['story_words']} words, {rec['finish_reason']}, "
                  f"~{rate * (len(jobs) - n) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
