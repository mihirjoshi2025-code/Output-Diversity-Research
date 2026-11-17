# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Count the paper's 11 Core words in text, or summarize JSONL files.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

CORE = {
    "names": ("elias", "mara", "elara"),
    "professions": (
        "keeper", "baker", "mayor", "clockmaker", "fisherman", "librarian", "conductor",
    ),
    "locations": ("lighthouse",),
}
WORDS = tuple(word for group in CORE.values() for word in group)
PATTERN = re.compile(r"\b(?:" + "|".join(WORDS) + r")\b", re.IGNORECASE)


def score_core(text: str) -> dict:
    """Score exactly the supplied text; pass the story, not its prompt/reasoning."""
    found = Counter(match.lower() for match in PATTERN.findall(text))
    counts = {word: found[word] for word in WORDS}
    word_count = len(text.split())
    hits = sum(counts.values())
    return {
        "has_core": hits > 0,
        "category_count": sum(any(counts[word] for word in group) for group in CORE.values()),
        "word_count": word_count,
        "counts": counts,
        "core_ppm": hits * 1_000_000 / word_count if word_count else None,
    }


def story_only(text: str) -> str:
    """Remove a leading Think trace; don't silently count an unfinished trace."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    if "<think>" in text:
        raise ValueError("Unfinished <think> block: inspect this response before scoring it.")
    return text


def summarize(path: Path, field: str, strip_thinking: bool) -> dict:
    counts = Counter()
    documents_with = Counter()
    documents = core_documents = words = empty_documents = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = record[field]
                if not isinstance(text, str):
                    raise ValueError(f"{field!r} must be a string")
                if strip_thinking:
                    text = story_only(text)
                result = score_core(text)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            documents += 1
            core_documents += result["has_core"]
            words += result["word_count"]
            empty_documents += not text.strip()
            counts.update(result["counts"])
            documents_with.update(word for word, count in result["counts"].items() if count)
    return {
        "file": str(path),
        "field": field,
        "strip_thinking": strip_thinking,
        "records": documents,
        "empty_records": empty_documents,
        "records_with_core": core_documents,
        "percent_with_core": 100 * core_documents / documents if documents else None,
        "word_count": words,
        "core_ppm": sum(counts.values()) * 1_000_000 / words if words else None,
        "per_word": {
            word: {
                "occurrences": counts[word],
                "records_with_word": documents_with[word],
                "ppm": counts[word] * 1_000_000 / words if words else None,
            }
            for word in WORDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    text_parser = sub.add_parser("text", help="Score one UTF-8 text file")
    text_parser.add_argument("file", type=Path)
    text_parser.add_argument("--strip-thinking", action="store_true")
    jsonl_parser = sub.add_parser("jsonl", help="Summarize each JSONL file separately (no deduplication)")
    jsonl_parser.add_argument("paths", nargs="+", type=Path, help="JSONL files or directories containing them")
    jsonl_parser.add_argument("--field", default="content", help="Text field: content for generations, text for training extracts")
    jsonl_parser.add_argument("--strip-thinking", action="store_true")
    args = parser.parse_args()
    if args.command == "text":
        text = args.file.read_text(encoding="utf-8")
        result = score_core(story_only(text) if args.strip_thinking else text)
    else:
        files = sorted({file for path in args.paths for file in (path.glob("*.jsonl") if path.is_dir() else [path])})
        if not files:
            parser.error("No JSONL files found; unzip the data first.")
        result = [summarize(path, args.field, args.strip_thinking) for path in files]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
