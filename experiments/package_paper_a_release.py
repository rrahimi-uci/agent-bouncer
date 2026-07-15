#!/usr/bin/env python3
"""Stage a public Paper A v2 release from an explicit no-raw-prompt allowlist."""
from __future__ import annotations

import argparse
import pathlib
import shutil
import tempfile

try:  # package import in tests / direct script execution from repository root
    from . import paper_a_common as C
except ImportError:  # pragma: no cover - exercised by the Makefile script path
    import paper_a_common as C


ANALYSIS_FILES = (
    "results.json",
    "claim_checks.json",
    "seed_values.csv",
    "per_benchmark.csv",
    "sensitivity.json",
    "analysis_metadata.json",
)
ANALYSIS_TREES = {
    "tables": {".tex"},
    "figures": {".pdf", ".svg", ".png"},
    "composition": {".json", ".md"},
}
FORBIDDEN_NAMES = {
    "manifests",
    "runs",
    "base_scores",
    "audit",
    "smoke",
    "adapters",
    "raw",
}


def _copy_regular_file(source: pathlib.Path, destination: pathlib.Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise C.ArtifactContractError(
            f"release allowlist source is missing, non-regular, or symlinked: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def _copy_tree(source: pathlib.Path, destination: pathlib.Path, suffixes=None) -> None:
    if not source.is_dir() or source.is_symlink():
        raise C.ArtifactContractError(
            f"release allowlist tree is missing or symlinked: {source}")
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise C.ArtifactContractError(f"release source tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise C.ArtifactContractError(
                f"release source tree contains a non-regular entry: {path}")
        if suffixes is not None and path.suffix.lower() not in suffixes:
            raise C.ArtifactContractError(
                f"release analysis tree contains an unexpected file type: {path}")
        _copy_regular_file(path, destination / path.relative_to(source))
        copied += 1
    if copied == 0:
        raise C.ArtifactContractError(f"release allowlist tree is empty: {source}")


def _assert_public_shape(stage_root: pathlib.Path, artifact_rel: pathlib.Path) -> None:
    artifact = stage_root / artifact_rel
    for path in stage_root.rglob("*"):
        if path.is_symlink():
            raise C.ArtifactContractError(f"staged release contains a symlink: {path}")
        if any(part.lower() in FORBIDDEN_NAMES for part in path.relative_to(stage_root).parts):
            raise C.ArtifactContractError(
                f"staged release contains a forbidden raw/full-artifact path: {path}")
    expected_top = {"LOCK.json", "RELEASE.json", "public_manifests", "scores"}
    observed_top = {path.name for path in artifact.iterdir()}
    if not expected_top.issubset(observed_top):
        raise C.ArtifactContractError(
            f"staged release is missing required entries: {sorted(expected_top - observed_top)}")
    score_names = {path.name for path in (artifact / "scores").iterdir()}
    if score_names != {"scores.parquet", "metadata.json"}:
        raise C.ArtifactContractError(
            f"staged scores directory violates the exact allowlist: {sorted(score_names)}")


def stage_release(
    artifact_root: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    repo_root: str | pathlib.Path | None = None,
    force: bool = False,
) -> pathlib.Path:
    """Verify the trust chain, then atomically stage only public release files."""
    repo = pathlib.Path(repo_root or C.REPO_ROOT).resolve()
    source = pathlib.Path(artifact_root).resolve()
    destination = pathlib.Path(output).resolve()
    try:
        artifact_rel = source.relative_to(repo)
    except ValueError as exc:
        raise C.ArtifactContractError(
            f"artifact root must be inside the repository: {source}") from exc
    if destination == source or source in destination.parents:
        raise C.ArtifactContractError(
            "release output must not be the artifact root or one of its descendants")
    if destination.is_symlink():
        raise C.ArtifactContractError("release output must not be a symlink")
    if destination.exists() and not force:
        raise FileExistsError(f"release output already exists: {destination}")

    lock = C.load_lock(source / "LOCK.json", verify_files=False, repo_root=repo)
    canonical_source = C.resolved_path(C.artifact_paths(lock)["root"], repo)
    if source != canonical_source:
        raise C.ArtifactContractError(
            f"artifact root differs from the lock-authoritative root: {canonical_source}")
    C.verify_release_cache_lock(lock, repo_root=repo)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper-a-release-", dir=destination.parent) as temp:
        stage = pathlib.Path(temp) / "package"
        artifact_out = stage / artifact_rel
        _copy_regular_file(source / "LOCK.json", artifact_out / "LOCK.json")
        _copy_regular_file(source / C.RELEASE_FILENAME, artifact_out / C.RELEASE_FILENAME)
        _copy_tree(source / "public_manifests", artifact_out / "public_manifests")
        _copy_regular_file(
            source / "scores" / "scores.parquet",
            artifact_out / "scores" / "scores.parquet",
        )
        _copy_regular_file(
            source / "scores" / "metadata.json",
            artifact_out / "scores" / "metadata.json",
        )
        analysis = source / "analysis"
        if analysis.is_dir():
            for name in ANALYSIS_FILES:
                path = analysis / name
                if path.exists():
                    _copy_regular_file(path, artifact_out / "analysis" / name)
            for tree, suffixes in ANALYSIS_TREES.items():
                path = analysis / tree
                if path.exists():
                    _copy_tree(path, artifact_out / "analysis" / tree, suffixes)
        anchor_source = repo / C.DEFAULT_RELEASE_ANCHOR_PATH
        _copy_regular_file(anchor_source, stage / C.DEFAULT_RELEASE_ANCHOR_PATH)
        _assert_public_shape(stage, artifact_rel)

        checksum_lines = []
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            rel = path.relative_to(stage).as_posix()
            checksum_lines.append(f"{C.sha256_file(path)}  {rel}")
        (stage / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

        if destination.exists():
            shutil.rmtree(destination)
        stage.rename(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="canonical artifacts/paper_a_sft_v2 root")
    parser.add_argument("--out", required=True, help="new staging directory outside the repo")
    parser.add_argument("--force", action="store_true", help="replace an existing staging directory")
    args = parser.parse_args()
    out = stage_release(args.root, args.out, force=args.force)
    print(f"staged public Paper A v2 release: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
