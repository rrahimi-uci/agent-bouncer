"""Bundle the cached benchmark subsets into notebooks/data/ so the notebooks/ folder is a standalone,
self-contained unit (copy/zip it and the eval sets travel with the notebook). Excludes safepyramid (48 MB;
the SmolLM3 notebook does not use it). Run: python scripts/bundle_notebook_data.py"""
from __future__ import annotations

import shutil
from pathlib import Path

# the 7 benchmarks the SmolLM3 notebook evaluates on (positive class = unsafe)
BENCHES = ["beavertails", "openai_moderation", "toxicchat", "prompt_injections",
           "jailbreak_classification", "jailbreakbench", "xstest"]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "benchmarks"
DST = ROOT / "notebooks" / "data" / "benchmarks"


def copy_set(subdir: str) -> int:
    src = SRC / subdir if subdir else SRC
    dst = DST / subdir if subdir else DST
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for b in BENCHES:
        f = src / f"{b}.jsonl"
        if f.exists():
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"no benchmark cache at {SRC}")
    # Bundle only the full sets; the notebook derives class-balanced matched-n eval subsets from them
    # (and uses them for the offline train fallback), so shipping the pre-cut subsets too would be redundant.
    n_full = copy_set("full")        # -> notebooks/data/benchmarks/full/*.jsonl
    total = sum(f.stat().st_size for f in DST.rglob("*.jsonl"))
    print(f"bundled {n_full} full benchmark sets into {(DST / 'full').relative_to(ROOT)} "
          f"({total/1e6:.1f} MB; safepyramid excluded)")


if __name__ == "__main__":
    main()
