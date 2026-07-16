# Unified report — build status (autonomous session)

Snapshot of what is done / running / pending for the merged report, written during the autonomous
window. Nothing here fabricates numbers; pending pieces await their locked runs.

## Done and committed
- **Plan + pre-registration.** `docs/unified-report-plan.md` (final, staged minimal-honest program +
  reproducibility contract + reproduce-function spec) and `docs/paper-c-prereg.md` (objective-axis
  hypotheses, the GRPO single-token null, H2 — committed *before* any DPO/GRPO run, so git history is
  the anti-HARKing timestamp).
- **Report scaffold that builds.** `unified_report.tex` → 4-act accessible article (Specialize /
  Objective axis / Compose / four high-compliance Domains), reusing the simplified-edition teaching
  boxes + `float` + `draftwarning`. Existing results (`Act I` SFT, `Act III` composition, `Act IV`
  mortgage) are `\input` from committed generated tables with verified numbers; the objective axis and
  the ExpGuard breadth are marked \textsc{pending}. Builds clean (no undefined refs/citations).
- **Reproduce harness.** `reproduce.py` + `make reproduce` — one entry point regenerating every table
  from committed per-row scores (Paper A pinned-env, Paper B, mortgage, ExpGuard `--from-scores`,
  Paper C when trained); `--check` asserts byte-identity.
- **ExpGuard eval code.** `experiments/eval_expguard_external.py` — scores the 4 checkpoints on ExpGuard
  (finance/health/law) via the canonical guard head; commits only text-free per-row scores.
- **Paper C code (ready; GPU-validation pending).** `experiments/paper_c_preference.py` (offline
  self-test passes) + `experiments/run_paper_c_objective.py` (imports clean; TRL DPO/GRPO/KTO/ORPO
  trainers, reusing the frozen rows + LoRA recipe + run_meta).

## Done and committed (cont.)

- **ExpGuard base eval — COMPLETE (Act IV breadth).** All 4 checkpoints scored zero-shot on ExpGuard
  (2,275 rows; finance/health/law) on a spot L4 GPU and independently re-run on Apple MPS (agree to
  3–4 decimals). Overall AP: SmolLM3-3B 0.956, Qwen3-4B 0.951, Qwen2.5-1.5B 0.921, SmolLM2-1.7B 0.883 —
  the best domain guard is *not* the largest model. `tab:expguard` + `fig:expguard-domains` are in Act IV;
  text-free per-row scores committed; `reproduce.py --check` passes. NB: the eval now stores the raw
  decision margin (not the saturating sigmoid), which is what makes AP reproducible from committed scores.
  The *tuned* (base-vs-SFT) ExpGuard comparison remains future work.

## Pending (needs a GPU; won't finish in this window)
- **Objective axis (DPO/GRPO), Act II.** Code is ready and offline-validated; the actual training needs
  a GPU + the pinned CUDA stack + the (non-gated) training-source fetch. Per the plan's
  "validate before scaling", the first step is a single `--smoke` cell, then the full run.

## GPU launch runbook (objective axis)
Recommended: 1× A100-40GB **spot** (`a2-highgpu-1g`, project `jazzx-gcp-poc-1`; L4 `g2-standard-8` as
fallback). `gcloud` is at `~/google-cloud-sdk/bin/gcloud`, authed as `reza.rahimi@jazzx.ai`.
1. Provision the spot VM; install the pinned stack (`requirements.txt`: torch+cu, transformers 5.12.1,
   trl 1.7.0, peft 0.19.1, accelerate 1.14.0); clone the repo at the committed SHA.
2. **Validate one cell:** `python experiments/run_paper_c_objective.py --lock <LOCK> --objective dpo
   --model-key qwen25_15b --seed 42 --out-dir /tmp/smoke --smoke` → expect `status: smoke`, an adapter,
   and a `run_meta.json`. Then score it with `eval_paper_a_sft` and confirm a sane AP. Reconfirm the TRL
   DPO/GRPO trainer signatures against the pinned trl 1.7.0 (the one place API drift could bite).
3. **Launch the full run** only if the smoke is clean: {dpo, grpo} × 4 checkpoints × 5 seeds = 40
   adapters into `artifacts/paper_c_objective_v2/runs/...`; score each; `analyze_paper_c.py` emits the
   Act II table; commit the text-free per-row scores.
4. Cost guard: ~15–18 A100-spot GPU-hours (~$20–25). Training is checkpoint-restartable (spot-safe).

## Why the GPU run was not launched unattended
The full run (~15+ GPU-hr) cannot finish in the 2-hour window regardless, and launching 40 jobs of
new trainer code that has not been GPU-validated risks wasted budget and non-reproducible science —
against the report's own defensibility bar. Everything is staged so the run can be launched behind a
single validated `--smoke` cell.
