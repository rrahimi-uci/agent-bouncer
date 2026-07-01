# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial repository scaffold: taxonomy, verdict schema, reference `KeywordGuard`,
  verifiable reward functions, metrics (incl. false-positive-on-benign), MLflow eval
  harness, CLI, and stubs for SFT/GRPO training and standard-benchmark adapters.
- Data unification (roadmap phase 1): offline-tested normalizers mapping WildGuardMix,
  BeaverTails, Aegis 2.0, and XSTest onto the unified taxonomy; deterministic
  train/validation split and JSONL I/O; wired `scripts/download_data.py`.
