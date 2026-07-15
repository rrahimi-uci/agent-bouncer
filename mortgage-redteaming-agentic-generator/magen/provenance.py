"""Content hashing, near-dup families, and decontamination.

Reuses the parent repo's canonical `guard_research.provenance` (NFKC SHA-256 + a pinned
NumPy MinHash) so family assignments match Paper A exactly. A small pure-Python fallback is
used only if guard_research is not importable, so the offline demo still runs; the fallback
is clearly flagged and must not be used for a release build.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

try:  # canonical path
    from guard_research.provenance import (  # type: ignore
        normalize_text, content_sha256, minhash_signature, estimated_jaccard,
        MINHASH_JACCARD_THRESHOLD,
    )
    BACKEND = "guard_research"
except Exception:  # pragma: no cover - fallback only when the lib is absent
    BACKEND = "fallback-blake2b"
    MINHASH_JACCARD_THRESHOLD = 0.85
    _WS = re.compile(r"\s+")

    def normalize_text(t: str) -> str:
        import unicodedata
        return _WS.sub(" ", unicodedata.normalize("NFKC", str(t)).lower()).strip()

    def content_sha256(t: str) -> str:
        return hashlib.sha256(normalize_text(t).encode("utf-8")).hexdigest()

    def _shingles(norm: str, ngram: int = 5) -> set[str]:
        toks = norm.split()
        if len(toks) < ngram:
            return {norm} if norm else set()
        return {" ".join(toks[i:i + ngram]) for i in range(len(toks) - ngram + 1)}

    def minhash_signature(text: str, num_perm: int = 64, ngram: int = 5):
        sh = _shingles(normalize_text(text), ngram)
        if not sh:
            return tuple([0] * num_perm)
        sig = []
        for p in range(num_perm):
            best = min(int.from_bytes(
                hashlib.blake2b(f"{p}:{s}".encode(), digest_size=8).digest(), "big")
                for s in sh)
            sig.append(best)
        return tuple(sig)

    def estimated_jaccard(a, b) -> float:
        if not a or not b:
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def hash_rows(rows: list[Any]) -> None:
    """Set content_sha256 on each row in-place."""
    for r in rows:
        r.content_sha256 = content_sha256(r.user_prompt)


def dedup_exact(rows: list[Any]) -> tuple[list[Any], int]:
    """Drop later rows sharing a content_sha256. Returns (kept, n_dropped)."""
    seen: set[str] = set()
    kept: list[Any] = []
    for r in rows:
        if r.content_sha256 in seen:
            continue
        seen.add(r.content_sha256)
        kept.append(r)
    return kept, len(rows) - len(kept)


def assign_content_families(rows: list[Any], threshold: float | None = None) -> int:
    """Cluster rows into near-dup families (union-find over MinHash Jaccard >= threshold);
    write `content_family` on each row. Returns the family count.

    O(n^2) in the signature comparison — fine for benchmark-scale sets (<~50k). For larger
    sets, LSH banding would replace the pairwise loop.
    """
    thr = MINHASH_JACCARD_THRESHOLD if threshold is None else threshold
    sigs = [minhash_signature(r.user_prompt) for r in rows]
    uf = _UnionFind(len(rows))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if estimated_jaccard(sigs[i], sigs[j]) >= thr:
                uf.union(i, j)
    roots: dict[int, str] = {}
    for i, r in enumerate(rows):
        root = uf.find(i)
        cf = roots.setdefault(root, f"CF-{len(roots):05d}")
        r.content_family = cf
    return len(roots)


def _load_general_hashes(index_dir: str) -> set[str]:
    """Best-effort: collect content hashes from the parent repo's text-free general index.

    The public index is text-free, so we can exact-hash-decontaminate against it. Near-dup
    cross-source decontam additionally needs the source text or precomputed signatures; when
    those are unavailable we honestly report exact-only coverage.
    """
    import json
    import os
    hashes: set[str] = set()
    if not index_dir or not os.path.isdir(index_dir):
        return hashes
    for base, _dirs, files in os.walk(index_dir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            try:
                obj = json.load(open(os.path.join(base, fn)))
            except Exception:
                continue
            for h in _walk_hashes(obj):
                hashes.add(h)
    return hashes


_HEXRE = re.compile(r"^[0-9a-f]{64}$")


def _walk_hashes(obj: Any):
    """Yield any 64-hex-char string values found anywhere in a nested JSON object."""
    if isinstance(obj, str):
        if _HEXRE.match(obj):
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_hashes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_hashes(v)


def decontaminate(rows: list[Any], general_sources_index: str) -> dict[str, Any]:
    """Drop rows whose content hash collides with the general sources index (exact).

    Mutates `rows`? No — returns (kept_rows, report). Callers replace their list.
    """
    general = _load_general_hashes(general_sources_index)
    kept, dropped = [], []
    for r in rows:
        if general and r.content_sha256 in general:
            dropped.append(r.id)
        else:
            kept.append(r)
    return {
        "kept": kept,
        "report": {
            "backend": BACKEND,
            "general_index": general_sources_index,
            "general_hashes_loaded": len(general),
            "exact_cross_hits_dropped": len(dropped),
            "dropped_ids": dropped,
            "neardup_cross_source": "exact-only; near-dup cross-source decontam requires "
                                    "the general source text or precomputed signatures",
        },
    }
