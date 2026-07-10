"""
demo_factdedup_check.py
========================
Shows BEFORE (per-section decomposition, no cross-section merging) vs AFTER
(dedupe_across_sections applied) fact lists for one real note, side by side,
so the embedding-based cross-section dedup in atomic_luq.py can be manually
verified rather than trusted blind.

Usage:
    python demo_factdedup_check.py --sample 5 --note 0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import atomic_luq as al

GEN_DIR = Path("luq_out/llama/generations/aci/test2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--note", type=int, default=0)
    args = ap.parse_args()

    gen = json.loads((GEN_DIR / f"sample_{args.sample:03d}_generations.json").read_text())
    note = gen["notes"][args.note]
    sections = al.parse_sections(note)

    before = {}
    for sec in al.SECTION_NAMES:
        text = sections.get(sec, "")
        before[sec] = al.decompose_section(text) if text else []

    n_before = sum(len(v) for v in before.values())
    after = al.dedupe_across_sections(before)
    n_after = sum(len(v) for v in after.values())

    print(f"BEFORE dedup: {n_before} facts")
    for sec in al.SECTION_NAMES:
        for f in before[sec]:
            print(f"  [{sec}] {f}")

    print()
    print(f"AFTER dedup: {n_after} facts  ({n_before - n_after} removed)")
    for sec in al.SECTION_NAMES:
        for f in after[sec]:
            print(f"  [{sec}] {f}")

    print()
    # Facts present in BEFORE but missing from AFTER -- these are what the
    # dedup step decided were duplicates. Print them explicitly so it's easy
    # to judge by eye whether each one is a genuine duplicate or a false merge.
    before_flat = [f for sec in al.SECTION_NAMES for f in before[sec]]
    after_flat = [f for sec in al.SECTION_NAMES for f in after[sec]]
    removed = list(before_flat)
    for f in after_flat:
        if f in removed:
            removed.remove(f)

    print(f"REMOVED as \"duplicates\" ({len(removed)}):")
    for f in removed:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
