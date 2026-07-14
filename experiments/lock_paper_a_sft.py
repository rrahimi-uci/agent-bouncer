#!/usr/bin/env python
"""Create artifacts/paper_a_sft_v2/LOCK.json (plan sec 14.1).

The lock is created AFTER manifests/tests/smoke validation and BEFORE final
training. It freezes every input to the fixed-panel base-vs-LoRA-SFT study so
that final-evaluation code can refuse an absent or mismatched lock.

Records (plan sec 14.1):
  git sha + dirty-state policy; model/tokenizer revisions; data revisions and
  manifest hashes; source inclusions/exclusions; license branch; prompt
  template + hash (model-independent spec hash plus per-checkpoint rendered
  template hashes and decision token ids); training recipe; seeds; metrics;
  target FPR + confidence method; primary contrasts; precision-focused analysis
  mode; statistical resampling rules; table/figure specs; failure handling;
  artifact paths.

Refuses to overwrite an existing lock without --force.

Usage:
  python experiments/lock_paper_a_sft.py \
    --config configs/paper_a_sft.yaml \
    --manifest artifacts/paper_a_sft_v2/manifests/manifest.json \
    --audit artifacts/paper_a_sft_v2/audit/audit.json \
    --out artifacts/paper_a_sft_v2/LOCK.json
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import pathlib

# path bootstrap so `import guard_research` / sibling module both resolve
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402

MANIFEST_FILES = list(C.LOCK_MANIFEST_FILES)


def _obj_sha256(obj) -> str:
    return C.canonical_obj_sha256(obj)


def resolve_models(config: dict) -> dict:
    """Model/tokenizer revisions: config['models'] is authoritative; fall back to panel."""
    models = {}
    cfg_models = config.get("models") or {}
    for key in C.MODEL_KEYS:
        panel = C.MODEL_PANEL[key]
        cm = cfg_models.get(key, {}) if isinstance(cfg_models, dict) else {}
        model_id = cm.get("model_id", panel["model_id"])
        rev = cm.get("model_revision", cm.get("revision", panel["revision"]))
        tok_rev = cm.get("tokenizer_revision", rev)
        models[key] = {
            "model_id": model_id,
            "model_revision": rev,
            "tokenizer_revision": tok_rev,
            "dtype": cm.get("dtype", config.get("dtype", "bfloat16")),
            "attn_implementation": cm.get("attn_implementation", config.get("attn_implementation")),
            "trust_remote_code": cm.get("trust_remote_code", True),
            "revision_source": "config" if key in cfg_models else "panel_default",
        }
    return models


def probe_tokenizers(models: dict, require: bool) -> dict:
    """Load each tokenizer at its pinned revision and freeze decision tokens +
    rendered-template hash (plan sec 7). Degrades gracefully if models are absent."""
    out = {}
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        if require:
            raise
        return {k: {"status": "transformers_unavailable", "error": str(e)} for k in models}
    for key, m in models.items():
        rec = {"status": "unavailable"}
        try:
            tok = AutoTokenizer.from_pretrained(
                m["model_id"], revision=m["tokenizer_revision"],
                trust_remote_code=m.get("trust_remote_code", True))
            dt = C.resolve_decision_tokens(tok)
            rec = {
                "status": "ok",
                "safe_str": dt["safe_str"], "unsafe_str": dt["unsafe_str"],
                "safe_token_id": dt["safe_id"], "unsafe_token_id": dt["unsafe_id"],
                "prompt_template_sha256": C.template_sha256(tok),
            }
        except Exception as e:  # missing model / gated / no single-token convention
            rec = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            if require:
                raise RuntimeError(f"tokenizer probe failed for {key}: {e}") from e
        out[key] = rec
    return out


def build_lock(args) -> dict:
    development_override = bool(getattr(args, "development_override", False))
    config = C.load_config(args.config)
    prompt = C.prompt_identity()
    models = resolve_models(config)
    issues = []

    def require(condition, message):
        if condition:
            return
        if development_override:
            issues.append(message)
        else:
            raise C.ArtifactContractError(message)

    for model_key, model in models.items():
        require(isinstance(model.get("model_revision"), str)
                and re.fullmatch(C.FULL_COMMIT_SHA_RE, model["model_revision"]) is not None,
                f"{model_key} model_revision must be a full 40-hex commit SHA")
        require(isinstance(model.get("tokenizer_revision"), str)
                and re.fullmatch(C.FULL_COMMIT_SHA_RE, model["tokenizer_revision"]) is not None,
                f"{model_key} tokenizer_revision must be a full 40-hex commit SHA")
        require(model.get("dtype") in C.SUPPORTED_TORCH_DTYPES,
                f"{model_key} dtype must be one of {C.SUPPORTED_TORCH_DTYPES}")

    try:
        config_rel = pathlib.Path(args.config).resolve().relative_to(C.REPO_ROOT).as_posix()
    except ValueError:
        config_rel = None
    require(config_rel == "configs/paper_a_sft.yaml",
            "final lock config must be the tracked configs/paper_a_sft.yaml")

    out_arg = getattr(args, "out", None) or C.DEFAULT_ARTIFACTS_V2["lock"]
    artifact_root = (getattr(args, "artifact_root", None)
                     or os.path.dirname(os.fspath(out_arg))
                     or C.DEFAULT_ARTIFACTS_V2["root"])
    artifact_paths = C.artifact_paths_for_root(artifact_root)

    def resolved(path):
        value = pathlib.Path(path)
        return value.resolve() if value.is_absolute() else (C.REPO_ROOT / value).resolve()

    require(not C.path_is_within(artifact_root, C.DEFAULT_ARTIFACTS["root"]),
            "v2 lock may not use or nest inside the historical artifacts/paper_a_sft namespace")
    require(C.path_is_within(artifact_root, C.REPO_ROOT),
            "final v2 artifact root must resolve inside the repository")
    require(resolved(out_arg) == resolved(artifact_paths["lock"]),
            "--out must be <artifact-root>/LOCK.json")

    # -- manifests: hash the index + each split file that exists --
    manifests_dir = args.manifests_dir or (
        os.path.dirname(args.manifest) if args.manifest else artifact_paths["manifests"])
    require(resolved(manifests_dir) == resolved(artifact_paths["manifests"]),
            "v2 manifests must live under the bound v2 artifact root")
    manifests = {"dir": manifests_dir, "index": None, "splits": {}}
    if args.manifest and os.path.exists(args.manifest):
        manifests["index"] = {"path": args.manifest, "sha256": C.sha256_file(args.manifest)}
    require(manifests["index"] is not None,
            "final lock requires the manifest index supplied by --manifest")
    for fn in MANIFEST_FILES:
        p = os.path.join(manifests_dir, fn)
        if os.path.exists(p):
            manifests["splits"][fn] = {"path": p, "sha256": C.sha256_file(p),
                                       "rows": sum(1 for _ in open(p, "r", encoding="utf-8") if _.strip())}
        else:
            manifests["splits"][fn] = {"path": p, "sha256": None, "rows": None, "missing": True}
            require(False, f"final lock requires manifest split: {p}")
    train_manifest_sha256 = manifests["splits"].get("train.jsonl", {}).get("sha256")

    # -- audit / power provenance --
    audit = None
    audit_obj = None
    if args.audit and os.path.exists(args.audit):
        require(resolved(os.path.dirname(args.audit)) == resolved(artifact_paths["audit"]),
                "v2 audit must live under the bound v2 artifact root")
        audit = {"path": args.audit, "sha256": C.sha256_file(args.audit)}
        audit_obj = C.read_json(args.audit)
        require(audit_obj.get("all_hard_assertions_pass") is True,
                "final lock requires audit all_hard_assertions_pass=true")
        hard = audit_obj.get("hard_assertions")
        require(isinstance(hard, dict) and hard and all(v is True for v in hard.values()),
                "final lock requires a nonempty, fully passing hard-assertion set")
        require(audit_obj.get("audit_contract_version") == C.AUDIT_CONTRACT_VERSION,
                "final lock requires the current audit contract version")
        require(isinstance(hard, dict) and set(hard) == set(C.AUDIT_HARD_ASSERTION_KEYS),
                "final lock requires the exact hard-assertion schema")
        require((audit_obj.get("manifest_index") or {}).get("observed_sha256")
                == (manifests.get("index") or {}).get("sha256"),
                "audit manifest-index hash must match the manifest index bound by the lock")
        audit_files = audit_obj.get("file_integrity") or {}
        for filename, locked_split in manifests["splits"].items():
            stem = filename[:-len(".jsonl")]
            audited = audit_files.get(stem) or {}
            require(audited.get("sha256_matches") is True
                    and audited.get("row_count_matches") is True,
                    f"audit did not pass file integrity for {filename}")
            require(audited.get("observed_sha256") == locked_split.get("sha256"),
                    f"audit digest for {filename} differs from the manifest being locked")
            require(audited.get("observed_rows") == locked_split.get("rows"),
                    f"audit row count for {filename} differs from the manifest being locked")
    else:
        require(False, "final lock requires a present --audit artifact")
    public_manifest_path = os.path.join(artifact_paths["public_manifests"], "manifest.json")
    public_release = None
    if os.path.isfile(public_manifest_path):
        public_release = {
            "manifest_path": public_manifest_path,
            "manifest_sha256": C.sha256_file(public_manifest_path),
        }
        public_validation = ((audit_obj or {}).get("public_release_validation") or {})
        require(public_validation.get("ok") is True,
                "final lock requires a passing public-release audit")
        require(public_validation.get("manifest_sha256") == public_release["manifest_sha256"],
                "audit public-manifest digest differs from the release being locked")
    else:
        require(False, f"final lock requires public manifest: {public_manifest_path}")
    power = None
    seed_count_decision = None
    analysis_mode = args.analysis_mode or config.get("analysis_mode") or "precision_focused"
    if args.power:
        raise C.ArtifactContractError(
            "--power is disabled with powered-confirmatory mode; omit it for precision mode")
    if args.power and os.path.exists(args.power):
        preport = C.read_json(args.power)
        power = {"path": args.power, "sha256": C.sha256_file(args.power)}
        seed_count_decision = preport.get("seed_count_decision") or preport.get("decision")
        if args.analysis_mode is None and preport.get("analysis_mode"):
            analysis_mode = preport["analysis_mode"]
    if analysis_mode != "precision_focused":
        raise C.ArtifactContractError(
            "powered_confirmatory mode is disabled until a schema-bound power report and "
            "null-calibrated multiplicity implementation exist")

    git = C.git_provenance()
    execution_git = C.execution_git_provenance()
    require(execution_git.get("execution_clean") is True,
            "final lock requires clean tracked and untracked execution state: "
            f"dirty={execution_git.get('dirty_entries')} "
            f"missing_from_HEAD={execution_git.get('required_sources_missing_from_head')}")
    try:
        execution_sources = C.execution_source_hashes()
    except C.ArtifactContractError as exc:
        require(False, str(exc))
        execution_sources = {"files": {}, "aggregate_sha256": C.canonical_obj_sha256({})}

    recipe = copy.deepcopy(C.DEFAULT_RECIPE)
    # overlay config recipe values when present
    for k_cfg, k_lock in (("max_steps", "max_steps"), ("max_length", "max_length"),
                          ("learning_rate", "learning_rate"), ("warmup_ratio", "warmup_ratio"),
                          ("effective_batch", "effective_batch")):
        if k_cfg in config:
            recipe[k_lock] = config[k_cfg]
    if isinstance(config.get("lora"), dict):
        recipe["lora"].update({k: config["lora"][k] for k in ("r", "alpha", "dropout")
                               if k in config["lora"]})

    seeds = list(config.get("seeds", C.DEFAULT_SEEDS))
    data_seed = config.get("data_seed", C.DEFAULT_DATA_ORDER_SEED)
    data_order_seed = config.get("data_order_seed", C.DEFAULT_DATA_ORDER_SEED)
    target_fpr = float(config.get("target_fpr", C.DEFAULT_TARGET_FPR))
    reps = int(config.get("bootstrap_replicates", C.DEFAULT_BOOTSTRAP_REPLICATES))
    boot_seed = int(config.get("bootstrap_seed", C.DEFAULT_BOOTSTRAP_SEED))

    tok_probe = {}
    if args.probe_tokenizers:
        tok_probe = probe_tokenizers(
            models, require=(not development_override or args.require_tokenizer_probe))
        require(set(tok_probe) == set(C.MODEL_KEYS)
                and all(v.get("status") == "ok" for v in tok_probe.values()),
                "final lock requires successful tokenizer probes for all models")
    else:
        require(False, "final lock requires successful tokenizer probes for all models")

    lock = {
        "lock_contract_version": C.LOCK_CONTRACT_VERSION,
        "finalization_status": ("development_unverified" if development_override else "final"),
        "development_issues": issues,
        "schema_version": config.get("schema_version", 1),
        "study_id": config.get("study_id", "paper_a_sft"),
        "created_utc": C.utcnow(),
        "config": {"path": args.config, "sha256": C.sha256_file(args.config),
                   "obj_sha256": _obj_sha256(config)},
        "git": {**git, **execution_git,
                "dirty_state_policy": ("development_override" if development_override
                                       else "require_clean_execution_state")},
        "execution_sources": execution_sources,
        "license_branch": config.get("data_branch", config.get("license_branch",
                                                               "academic_noncommercial")),
        "data": {
            "data_seed": data_seed,
            "data_order_seed": data_order_seed,
            "train_sources": config.get(
                "train_sources", C.DEFAULT_DATA_CONTRACT["train_sources"]),
            "excluded_train_sources": config.get(
                "excluded_train_sources", C.DEFAULT_DATA_CONTRACT["excluded_train_sources"]),
            "rows_per_source": config.get(
                "rows_per_source", C.DEFAULT_DATA_CONTRACT["rows_per_source"]),
            "rows_per_source_label": config.get(
                "rows_per_source_label", C.DEFAULT_DATA_CONTRACT["rows_per_source_label"]),
        },
        "models": models,
        "tokenizer_probe": tok_probe,
        "prompt": {
            "version": C.PROMPT_VERSION,
            "prompt_spec_sha256": prompt["prompt_sha256"],
            "spec_source": prompt["source"],
            "system_prompt": prompt["system"],
            "per_model_template_sha256": {k: v.get("prompt_template_sha256")
                                          for k, v in tok_probe.items()},
        },
        "recipe": recipe,
        "seeds": seeds,
        "n_checkpoints": len(C.MODEL_KEYS),
        "n_seeds": len(seeds),
        "n_final_cells": len(C.MODEL_KEYS) * len(seeds),
        "metrics": {
            "canonical_module": "guard_research.metrics",
            "primary": ["tie_aware_average_precision_per_benchmark",
                        "macro_ap_represented", "macro_ap_transfer", "base_to_sft_delta"],
            "secondary": ["auroc", "tpr_at_calibration_target_fpr", "realized_fpr",
                          "brier", "log_loss", "orbench_benign_fpr", "harmbench_recall"],
            "primary_metric": config.get("primary_metric", "macro_average_precision"),
        },
        "operating_point": {
            **copy.deepcopy(C.DEFAULT_OPERATING_POINT),
            "target_fpr": target_fpr,
        },
        "regime_benchmarks": C.REGIME_BENCHMARKS,
        "primary_contrasts": {
            "unit": "per_checkpoint_base_vs_sft",
            "regimes": ["represented", "transfer"],
            "estimand": "theta = (mean_delta_represented, mean_delta_transfer)",
            "delta": "M_R(SFT_{b,r}) - M_R(base_b)",
            "aggregate": ("fixed_panel_mean over 4 checkpoints of "
                          "(mean over seeds M_R(SFT_b) - M_R(base_b))"),
        },
        "analysis_mode": analysis_mode,
        "power_report": power,
        "seed_count_decision": seed_count_decision,
        "confidence_method": "hierarchical_paired_poisson_family_bootstrap",
        "resampling_rules": {
            **copy.deepcopy(C.DEFAULT_RESAMPLING_RULES),
            "replicates": reps,
            "rng_seed": boot_seed,
        },
        "sensitivity": {
            "leave_one_transfer_benchmark_out": True,
            "leave_one_base_out": True,
            "sign_stable_definition": "every leave-one-out estimate shares the full aggregate sign",
        },
        "descriptive_criteria": {
            "represented": "one-sided 95% lower bound above zero",
            "transfer": ("one-sided 95% upper bound below zero with leave-one-transfer-"
                         "benchmark-out and leave-one-checkpoint-out sign stability"),
            "joint_pattern": "both descriptive criteria hold",
            "formal_rejection_authorized": False,
            "rq4": "descriptive_only",
        },
        "claim_gates": None,
        "tables": {
            "table3_primary": {"path": "analysis/tables/table3_primary.tex",
                               "content": "per-base represented/transfer base, SFT mean, "
                                          "delta+interval, 5 seed values, fixed-panel aggregate"},
            "table4_per_benchmark": {"path": "analysis/tables/table4_per_benchmark.tex",
                                     "content": "per-source paired delta, TPR@target FPR, "
                                                "realized FPR, OR-Bench benign FPR, HarmBench recall"},
        },
        "figures": {
            "specialization_plane": {"path": "analysis/figures/specialization_plane.pdf",
                                     "content": "x=represented delta, y=transfer delta; "
                                                "color per checkpoint; point per seed; "
                                                "zero lines; fixed-panel mean marker"},
        },
        "failure_handling": {
            "keep_failed_runs": True,
            "failed_cell_blocks_fixed_panel_aggregate": True,
            "require_all_cells_before_scoring": True,
            "cache_mismatch_policy": "recompute_never_trust_row_count",
            "no_feasible_threshold_is_reportable": True,
        },
        "audit": audit,
        "public_release": public_release,
        "manifests": manifests,
        "train_manifest_sha256": train_manifest_sha256,
        "artifact_paths": artifact_paths,
        "score_code_version": "paper_a_sft_scorer_v2",
        "analysis_code_version": "paper_a_sft_analysis_v2",
        "software_versions": C.software_versions(),
    }
    lock["lock_sha256"] = _obj_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
    C.verify_lock(lock, allow_development=development_override,
                  verify_files=not development_override)
    return lock


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write Paper A LOCK.json (plan sec 14.1).")
    ap.add_argument("--config", required=True, help="configs/paper_a_sft.yaml")
    ap.add_argument("--manifest", default=None, help="manifests/manifest.json index")
    ap.add_argument("--manifests-dir", default=None,
                    help="directory holding the split jsonl files (default: --manifest parent)")
    ap.add_argument("--audit", default=None, help="audit/audit.json")
    ap.add_argument("--power", default=None, help="design/power_report.json")
    ap.add_argument("--out", default=C.DEFAULT_ARTIFACTS_V2["lock"],
                    help="output LOCK.json path (defaults to the isolated v2 namespace)")
    ap.add_argument("--artifact-root", default=None,
                    help="strict v2 artifact root; --out must be ROOT/LOCK.json")
    ap.add_argument("--analysis-mode", default=None,
                    choices=["precision_focused"],
                    help="confirmatory mode is disabled pending a bound test/multiplicity schema")
    ap.add_argument("--force", action="store_true", help="overwrite an existing lock")
    ap.add_argument("--require-clean", action="store_true", default=True,
                    help="deprecated compatibility flag; final locks are clean by default")
    ap.add_argument(
        "--development-override", action="store_true",
        help=("write an explicitly development_unverified lock despite missing/dirty inputs; "
              "such a lock is rejected by final training/evaluation"),
    )
    probe = ap.add_mutually_exclusive_group()
    probe.add_argument("--probe-tokenizers", dest="probe_tokenizers", action="store_true",
                       default=True, help="load each tokenizer to freeze decision tokens (default)")
    probe.add_argument("--skip-tokenizer-probe", dest="probe_tokenizers", action="store_false",
                       help="do not load tokenizers (spec-level prompt hash only)")
    ap.add_argument("--require-tokenizer-probe", action="store_true",
                    help="fail if any tokenizer cannot be loaded/verified")
    args = ap.parse_args(argv)

    out = C.abspath(args.out) if not os.path.isabs(args.out) else args.out
    if os.path.exists(out) and not args.force:
        print(f"[lock] refusing to overwrite existing lock at {out} (use --force).",
              file=sys.stderr)
        return 2

    if not os.path.exists(args.config):
        print(f"[lock] config not found: {args.config}", file=sys.stderr)
        return 2

    try:
        lock = build_lock(args)
    except (C.ArtifactContractError, RuntimeError) as exc:
        print(f"[lock] refusing to create lock: {exc}", file=sys.stderr)
        return 2
    C.write_json(out, lock)
    probed = sum(1 for v in lock["tokenizer_probe"].values() if v.get("status") == "ok")
    print(f"[lock] wrote {out}")
    print(f"[lock]   contract=v{lock['lock_contract_version']} "
          f"status={lock['finalization_status']}")
    print(f"[lock]   git_sha={lock['git']['git_sha']} "
          f"execution_clean={lock['git']['execution_clean']}")
    print(f"[lock]   analysis_mode={lock['analysis_mode']} "
          f"cells={lock['n_final_cells']} seeds={lock['seeds']}")
    print(f"[lock]   prompt_spec_sha256={lock['prompt']['prompt_spec_sha256'][:16]}... "
          f"({lock['prompt']['spec_source']})")
    print(f"[lock]   train_manifest_sha256={str(lock['train_manifest_sha256'])[:16]}...")
    print(f"[lock]   tokenizers probed ok: {probed}/{len(lock['models'])}")
    print(f"[lock]   lock_sha256={lock['lock_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
