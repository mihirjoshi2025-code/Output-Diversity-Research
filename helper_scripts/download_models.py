# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==2.0.0"]
# ///
"""
List or download the OLMo 3 7B Think (and Instruct SFT / DPO) checkpoints for the project.
"""

import argparse
from pathlib import Path

MODELS = {
    "sft": ("allenai/Olmo-3-7B-Think-SFT", "6ff857587e040d6d523a3d5f3a56e918f5401d66"),
    "dpo": ("allenai/Olmo-3-7B-Think-DPO", "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38"),
    "rlvr": ("allenai/Olmo-3-7B-Think", "d97e442d7cc678210054dbcc9b440894d62c89a4"),
    "instruct-sft": ("allenai/Olmo-3-7B-Instruct-SFT", "e1452fc572d51966ff4aaeb25118b891eb93e549"),
    "instruct-dpo": ("allenai/Olmo-3-7B-Instruct-DPO", "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"),
}
FILES = [
    "*.safetensors", "model.safetensors.index.json", "config.json",
    "generation_config.json", "tokenizer*.json", "special_tokens_map.json",
    "*.jinja", "merges.txt", "vocab.json", "README.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Print the pinned checkpoints without downloading weights")
    download = sub.add_parser("download", help="Download selected checkpoints (about 14.6 GB each)")
    download.add_argument("stages", nargs="+", choices=tuple(MODELS))
    download.add_argument("--output-dir", type=Path, required=True, help="Model storage on your compute machine, not in the shared Drive folder")
    download.add_argument("--dry-run", action="store_true", help="Print destinations without downloading or creating directories")
    args = parser.parse_args()
    stages = list(MODELS) if args.command == "list" else list(dict.fromkeys(args.stages))
    for stage in stages:
        repo, revision = MODELS[stage]
        print(f"{stage}: https://huggingface.co/{repo}/tree/{revision}", flush=True)
        if args.command == "list":
            continue
        target = args.output_dir.expanduser() / repo.split("/")[1]
        print(f"  destination: {target}", flush=True)
        if not args.dry_run:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo, revision=revision, local_dir=target, allow_patterns=FILES)


if __name__ == "__main__":
    main()
