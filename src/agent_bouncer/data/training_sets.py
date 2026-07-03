"""Training-set construction **strategies** — the AI-engineering choices that decide
what a guard learns from, with train/test separation guaranteed *by construction*.

Each strategy pools data from one or more registered benchmarks, then splits every
source into disjoint train/test parts (`split.train_test_split`) and keeps the train
parts for the training set and the test parts as a held-out eval set — so a model can
never be scored on a prompt it trained on. Balanced ½ safe / ½ unsafe by default.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile

from agent_bouncer.data.split import assert_no_leakage, train_test_split

OUT_DIR = "data/train_sets"
_TRAINING_SET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


#: Strategy catalog surfaced to the UI/CLI.
#: Every strategy accepts **any number** of sources (1 → all available); the strategy only
#: decides how the drawn data is *composed*, not how many benchmarks you may pick.
STRATEGIES: dict[str, dict] = {
    "balanced": {
        "label": "Balanced",
        "desc": "½ safe / ½ unsafe from your chosen source(s) — the clean classification baseline.",
        "min_sources": 1, "max_sources": None,
    },
    "mixed": {
        "label": "Mixed sources (diversity)",
        "desc": "Blend the chosen datasets so the guard generalizes beyond one distribution.",
        "min_sources": 1, "max_sources": None,
    },
    "over_refusal_aware": {
        "label": "Over-refusal-aware",
        "desc": "Your chosen source(s) + extra benign-but-scary prompts (XSTest-style) so the "
                "guard learns not to over-block — directly targets FPR@benign.",
        "min_sources": 1, "max_sources": None,
    },
    "red_team": {
        "label": "Red-team hardening",
        "desc": "Prompt-injection + jailbreak sources so the guard learns to catch attacks, "
                "not just harmful content.",
        "min_sources": 1, "max_sources": None,
    },
}


def source_bounds_text(spec: dict) -> str:
    """Human-friendly source-count note for a strategy spec."""
    min_sources = int(spec["min_sources"])
    max_sources = spec.get("max_sources")
    if max_sources is None:
        return f"{min_sources}+ sources"
    if min_sources == max_sources:
        return f"exactly {min_sources} source{'s' if min_sources != 1 else ''}"
    return f"{min_sources}–{max_sources} sources"


def validate_strategy_sources(strategy: str, sources: list[str]) -> None:
    """Raise a clear error when a strategy gets the wrong number of sources."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {sorted(STRATEGIES)}")
    spec = STRATEGIES[strategy]
    min_sources = int(spec["min_sources"])
    max_sources = spec.get("max_sources")
    n_sources = len(sources)
    if n_sources < min_sources or (max_sources is not None and n_sources > max_sources):
        raise ValueError(f"{strategy!r} needs {source_bounds_text(spec)}, got {n_sources}")


def validate_training_set_name(name: str) -> str:
    """Return a safe training-set directory name or raise ``ValueError``.

    Names become subdirectories under ``data/train_sets``. Keep them as plain slugs so
    API/CLI callers cannot traverse outside that root or create ambiguous hidden paths.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("training set name is required")
    if os.path.isabs(cleaned) or "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
        raise ValueError("training set name must be a plain directory name")
    if not _TRAINING_SET_NAME_RE.fullmatch(cleaned):
        raise ValueError("training set name may contain only letters, numbers, '.', '_' and '-'")
    return cleaned


def _balanced(records: list[dict], per_class: int, rng: random.Random) -> list[dict]:
    safe = [r for r in records if r["label"] == "safe"]
    unsafe = [r for r in records if r["label"] == "unsafe"]
    rng.shuffle(safe)
    rng.shuffle(unsafe)
    k = min(per_class, len(safe), len(unsafe))
    out = safe[:k] + unsafe[:k]
    rng.shuffle(out)
    return out


def build_training_set(
    strategy: str,
    sources: list[str],
    *,
    name: str,
    per_class: int = 200,
    holdout_ratio: float = 0.2,
    seed: int = 42,
    loader=None,
) -> dict:
    """Build a train/test pair under ``data/train_sets/<name>/`` and return its metadata.

    ``loader(source)`` returns the full record list for a source (defaults to the
    benchmark registry loader). Every source is split disjointly *before* pooling, so
    the returned train and test sets are guaranteed leakage-free.
    """
    validate_strategy_sources(strategy, sources)
    name = validate_training_set_name(name)

    if loader is None:
        from agent_bouncer.evaluation.benchmarks import load_benchmark
        def loader(src):  # noqa: E306
            return load_benchmark(src)

    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    # per_class <= 0  ->  PERCENTAGE mode: use ALL pooled data, split X% test / (100-X)% train
    #                     (stratified, the typical ML train/test split).
    # per_class  > 0  ->  COUNT mode: a balanced subset of `per_class` per class, then a holdout.
    full_split = not per_class or per_class <= 0
    if full_split:
        from agent_bouncer.data.sampling import ratio_split
        pooled: list[dict] = []
        for src in sources:
            pooled += loader(src)
        train, test = ratio_split(pooled, test_ratio=holdout_ratio, stratified=True, seed=seed)
        aug_n = max(10, len(train) // 10)
    else:
        per_source_pc = max(1, per_class // max(1, len(sources)))
        for src in sources:
            recs = loader(src)
            tr, te = train_test_split(recs, test_ratio=holdout_ratio, seed=seed)
            train += _balanced(tr, per_source_pc, rng)
            test += _balanced(te, max(1, per_source_pc // 2), rng)
        aug_n = per_source_pc

    # over-refusal augmentation: add extra benign prompts to TRAIN only, excluding anything
    # already in train/test so the split stays leakage-free.
    if strategy == "over_refusal_aware":
        try:
            from agent_bouncer.data.split import _key
            seen = {_key(r) for r in train} | {_key(r) for r in test}
            extra = [r for r in loader("xstest") if r["label"] == "safe" and _key(r) not in seen]
            rng.shuffle(extra)
            train += extra[:aug_n]
        except Exception:  # noqa: BLE001 - augmentation is best-effort
            pass

    rng.shuffle(train)
    rng.shuffle(test)
    assert_no_leakage(train, test)  # guaranteed disjoint

    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    _write_jsonl(train, os.path.join(out, "train.jsonl"))
    _write_jsonl(test, os.path.join(out, "test.jsonl"))
    n_safe = sum(r["label"] == "safe" for r in train)
    meta = {
        "name": name, "strategy": strategy, "sources": sources, "seed": seed,
        "per_class": per_class, "holdout_ratio": holdout_ratio,
        "split_mode": "percentage" if full_split else "balanced_count",
        "test_pct": round(holdout_ratio * 100),
        "n_train": len(train), "n_test": len(test),
        "train_safe": n_safe, "train_unsafe": len(train) - n_safe,
        "train_path": os.path.join(out, "train.jsonl"),
        "test_path": os.path.join(out, "test.jsonl"),
        "leakage_checked": True,
        "preview": train[:12],
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({k: v for k, v in meta.items() if k != "preview"}, fh, indent=2)
    return meta


def list_training_sets() -> list[dict]:
    if not os.path.isdir(OUT_DIR):
        return []
    sets = []
    for name in sorted(os.listdir(OUT_DIR)):
        meta = os.path.join(OUT_DIR, name, "meta.json")
        if os.path.exists(meta):
            try:
                sets.append(json.load(open(meta)))
            except (ValueError, OSError):
                pass
    return sets


def _write_jsonl(records: list[dict], path: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
