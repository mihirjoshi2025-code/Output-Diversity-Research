# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""
Compare Jev and Qwen3Guard labels on the same (prompt, response) pairs, and both against WildChat's own flags.

Prints agreement and Cohen's kappa (3-way Safe/Controversial/Unsafe, plus two binary cuts), the confusion matrix,
each classifier's verdicts within each WildChat group, refusal and category agreement, and writes every
disagreement to data/wildchat/jev_qwen_disagreements.jsonl so the pairs can be read by hand.

    uv run helper_scripts/compare_safety_labels.py
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "data" / "wildchat"
LABELS = ["Safe", "Controversial", "Unsafe"]


def kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa: agreement corrected for what two raters would reach by chance."""
    po = (a == b).mean()
    pe = sum(a.eq(c).mean() * b.eq(c).mean() for c in set(a) | set(b))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, default=WC / "response_sample.jsonl")
    parser.add_argument("--qwen", type=Path, default=WC / "qwen_response_labels.jsonl")
    parser.add_argument("--jev", type=Path, default=WC / "jev_labels.jsonl")
    args = parser.parse_args()

    sample = pd.read_json(args.sample, lines=True)
    qwen = pd.read_json(args.qwen, lines=True)[["key", "safety", "categories", "refusal", "created"]]
    jev = pd.read_json(args.jev, lines=True)[["key", "safety", "category", "refusal", "safety_confidence",
                                              "model", "created"]]
    df = (sample.merge(qwen.add_prefix("qwen_").rename(columns={"qwen_key": "key"}), on="key")
                .merge(jev.add_prefix("jev_").rename(columns={"jev_key": "key"}), on="key"))
    unparsed = df.qwen_safety.isna().sum()
    df = df[df.qwen_safety.notna()]
    print(f"{len(df)} pairs labelled by both (sample {len(sample)}, Qwen unparsed {unparsed}); "
          f"Jev model {', '.join(sorted(df.jev_model.unique()))}\n")

    q, j = df.qwen_safety, df.jev_safety
    cuts = {
        "3-way (Safe / Controversial / Unsafe)": (q, j),
        "Unsafe vs not": (q.eq("Unsafe"), j.eq("Unsafe")),
        "Safe vs not (Controversial counts as not safe)": (q.eq("Safe"), j.eq("Safe")),
    }
    print("Jev vs Qwen3Guard")
    for name, (a, b) in cuts.items():
        print(f"  {name:48s} agree {100 * (a == b).mean():5.1f}%   kappa {kappa(a, b):.2f}")
    print("  (kappa: 0 = chance, 0.6-0.8 substantial, >0.8 near-perfect)\n")

    print("Confusion matrix (rows Qwen3Guard, columns Jev):")
    print(pd.crosstab(q, j).reindex(index=LABELS, columns=LABELS, fill_value=0).to_string(), "\n")

    print("Verdicts within each WildChat group (%):")
    for group, d in df.groupby("group"):
        t = pd.DataFrame({"Qwen3Guard": d.qwen_safety.value_counts(normalize=True),
                          "Jev": d.jev_safety.value_counts(normalize=True)}).reindex(LABELS).fillna(0)
        print(f"  {group} (n={len(d)})")
        print((100 * t).round(0).astype(int).to_string().replace("\n", "\n    ").join(["    ", ""]))
    print()

    both_flag = df[(q != "Safe") & (j != "Safe")]
    qcat = both_flag.qwen_categories.map(lambda c: c[0] if c else "None")
    print(f"Refusal agree {100 * (df.qwen_refusal == df.jev_refusal).mean():.1f}%  "
          f"| top category agree, where both say not Safe: {100 * (qcat == both_flag.jev_category).mean():.1f}% "
          f"(n={len(both_flag)})\n")

    dis = df[q != j]
    out = WC / "jev_qwen_disagreements.jsonl"
    dis[["key", "group", "qwen_safety", "qwen_categories", "jev_safety", "jev_category", "jev_safety_confidence",
         "response_openai_categories", "prompt", "response"]].to_json(out, orient="records", lines=True,
                                                                       force_ascii=False)
    print(f"{len(dis)} disagreements -> {out.relative_to(ROOT)} (read before trusting either; may be explicit)")
    for col in ["qwen_created", "jev_created"]:
        t = pd.to_datetime(df[col], format="%Y-%m-%dT%H:%M:%S%z")
        print(f"  {col.split('_')[0]:4s} wall time ~{(t.max() - t.min()).total_seconds():.0f}s for {len(df)} pairs")


if __name__ == "__main__":
    main()
