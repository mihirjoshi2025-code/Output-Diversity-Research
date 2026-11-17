# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Label prompts with a safety classifier (Qwen3Guard-Gen) running in llama-server.

Input is a JSONL file with one {"key": ..., "prompt": ...} per line. Each output line records the key,
the classifier's verdict (Safe / Controversial / Unsafe), its categories, the raw reply, and which model
and llama.cpp build produced it. Re-running with the same --out skips keys that are already saved, so a
stopped run can be resumed.

Start the server first, e.g.
    ~/src/llama.cpp/build/bin/llama-server -m ~/models/gguf/Qwen3Guard-Gen-4B-Q8_0.gguf --jinja \
        -c 32768 --parallel 4 --port 8081
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_stories import get, post  # noqa: E402

GUARD = ("Qwen/Qwen3Guard-Gen-4B", "6ec42827da0c1ff11e7a49dc269d2e810d27e108")  # converted to Q8_0 GGUF
MAX_CHARS = 12_000  # ~3K tokens; keeps every request inside one llama-server slot (8K tokens with -c 32768 -np 4)
SAFETY = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
CATEGORIES = re.compile(r"Categories:\s*(.+)")


def parse(reply: str) -> tuple[str | None, list[str]]:
    safety = SAFETY.search(reply)
    cats = CATEGORIES.search(reply)
    categories = [c.strip() for c in cats.group(1).split(",")] if cats else []
    return (safety.group(1) if safety else None), [c for c in categories if c and c != "None"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inp", type=Path, help="JSONL with key + prompt")
    parser.add_argument("--out", type=Path, required=True, help="JSONL output file (appended to)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests; match llama-server --parallel")
    parser.add_argument("--url", default="http://localhost:8081")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    props = get(f"{args.url}/props")
    server_model = props.get("model_path") or get(f"{args.url}/v1/models")["data"][0]["id"]
    if "qwen3guard" not in Path(server_model).name.lower():
        sys.exit(f"Server has {server_model!r} loaded, not Qwen3Guard. Restart llama-server with the guard GGUF.")
    server_build = props.get("build_info")

    done = set()
    if args.out.exists():
        lines = args.out.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):  # stopped mid-write: drop the partial line
            lines.pop()
            args.out.write_text("".join(lines))
        done = {json.loads(line)["key"] for line in lines}
    jobs = [rec for rec in map(json.loads, args.inp.open()) if rec["key"] not in done]
    print(f"{len(done)} already labelled, {len(jobs)} to go -> {args.out}", flush=True)

    def run(rec: dict) -> dict:
        prompt = rec["prompt"]
        body = {"messages": [{"role": "user", "content": prompt[:MAX_CHARS]}],
                "temperature": 0, "max_tokens": 64}
        resp = post(f"{args.url}/v1/chat/completions", body, args.timeout)
        choice = resp["choices"][0]
        reply = choice["message"].get("content") or ""
        safety, categories = parse(reply)
        return {
            "key": rec["key"], "safety": safety, "categories": categories, "raw": reply,
            "truncated": len(prompt) > MAX_CHARS, "finish_reason": choice.get("finish_reason"),
            "repo": GUARD[0], "revision": GUARD[1], "server_model": server_model,
            "llama_cpp_build": server_build, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    lock = threading.Lock()
    start, errors = time.time(), 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, rec) for rec in jobs]
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
            if n % 100 == 0 or n == len(jobs):
                rate = (time.time() - start) / n
                print(f"  [{n}/{len(jobs)}] ~{rate * (len(jobs) - n) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
