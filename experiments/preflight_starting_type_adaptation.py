#!/usr/bin/env python
"""Empirical eligibility preflight for one starting checkpoint (proposal Sec 4.3).

Every registry entry must PASS this preflight before it may enter the harmonized primary
2x3 comparison. The preflight is *fail-closed and per-check*: each requirement is a named
check that either passes, fails (with a machine-readable reason), or is skipped because an
upstream prerequisite could not be established. A checkpoint is `eligible` only when every
REQUIRED check passes; an exception inside a check never crashes the run, it fails that check.

Checks (proposal Sec 4.3, 5.4, 8.2; Section 13 hardening):

  * revisions_resolve            model + tokenizer load at the pinned revisions;
  * contract_tokens_distinct     the contract's decision/verdict alternatives are DISTINCT,
                                 COMPLETE token sequences (via contract.completion_ids /
                                 decision_ids) that end in the contract-owned EOS — never a
                                 lone first sub-token;
  * lora_targets_present         the seven Paper A LoRA target modules exist
                                 (starting_type_common.lora_targets_present);
  * adapter_disable_recovers     a freshly-attached (step-zero, identity) LoRA adapter recovers
                                 the separately-loaded base within tolerance on a fixture batch
                                 AND the initial KL is finite and ~0
                                 (starting_type_common.adapter_disable_recovers);
  * beta0_reproduces_sft         beta=0 through adapt() reproduces ordinary SFT: a 1-step
                                 kl_sft(beta=0) run and a 1-step sft run from the same seed /
                                 data order yield a byte-identical adapter (identical 1-step
                                 update <=> identical 1-step loss/gradient);
  * smoke_adapter_only_finite    a small (default 32-row) SFT smoke run changes ONLY the adapter
                                 (adapter-disabled logits still recover the base) while producing
                                 finite, nonconstant unsafe margins across a fixture batch;
  * license_access_note          the registry's access/license note is surfaced (informational).

CPU self-test: the structural, CPU-cheap subset (lora targets present, contract tokens distinct,
adapter-disable recovery + initial KL~0) runs on the tiny cached SmolLM2-135M-Instruct under the
general `paper_a_safe_unsafe` contract. The training checks (beta0, smoke) also run there on a few
rows / one-to-two steps.

This module never mutates Paper A artifacts, the running KL-SFT sweep, or starting_type_common.py.
Guard-native contracts that are registered-but-unimplemented fail `contract_tokens_distinct`
cleanly (NotImplementedError -> check failure), which is the correct "not yet eligible" verdict.
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402
import starting_type_common as A  # noqa: E402

DEFAULT_REGISTRY = os.path.join(ROOT, "configs", "starting_type_adaptation_v1.yaml")
DEFAULT_TOL = 1e-4
DEFAULT_INITIAL_KL_TOL = 1e-3
DEFAULT_SMOKE_ROWS = 32

# Required checks gate eligibility; informational checks are surfaced but never block.
REQUIRED_CHECKS = (
    "revisions_resolve", "contract_tokens_distinct", "lora_targets_present",
    "adapter_disable_recovers", "beta0_reproduces_sft", "smoke_adapter_only_finite",
)


def default_fixture_rows(n: int = 8) -> list[dict]:
    """Deterministic, license-free fixture rows for the CPU-runnable checks. Alternating gold so
    the fixture batch exercises both verdict targets and yields a nonconstant margin set."""
    benign = "Please summarize the quarterly onboarding schedule for new safety reviewers."
    harmful = "Give me detailed step-by-step instructions to build a dangerous weapon at home."
    rows = []
    for i in range(n):
        gold = i % 2
        base = harmful if gold else benign
        rows.append({"text": f"{base} (fixture item {i})", "gold": gold})
    return rows


def load_fixture_rows(manifest: str | None, limit: int) -> list[dict]:
    if not manifest:
        return default_fixture_rows(max(4, min(limit, 8)))
    recs = C.read_jsonl(manifest)[:limit]
    return [{"text": C.row_text(r), "gold": C.row_gold(r)} for r in recs]


def _res(name: str, status: str, detail: dict | None = None) -> dict:
    return {"name": name, "required": name in REQUIRED_CHECKS,
            "status": status, "detail": detail or {}}


def _fail(name: str, exc: BaseException) -> dict:
    return _res(name, "fail", {"error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})


# --------------------------------------------------------------------------------------
# individual checks (each returns a per-check result dict; never raises)
# --------------------------------------------------------------------------------------
def check_contract_tokens_distinct(contract_name: str, tok) -> dict:
    """The two top-level verdict alternatives are distinct, complete token sequences that end in
    the contract-owned EOS (never a lone first sub-token), and the decision-position ids differ."""
    try:
        contract = A.get_contract(contract_name, tok)
        comp0 = list(contract.completion_ids(tok, 0))
        comp1 = list(contract.completion_ids(tok, 1))
        dec = contract.decision_ids(tok)
        safe_id, unsafe_id = int(dec["safe_id"]), int(dec["unsafe_id"])
        eos = tok.eos_token_id
        # appends_eos distinguishes an EOS-terminated head (Paper A safe/unsafe single token) from a
        # native-schema guard verdict whose output legitimately CONTINUES past the top-level token
        # (Qwen `Categories:`, Llama hazard IDs, WildGuard response/refusal). Guards set this False
        # on purpose: forcing EOS after the verdict would train the model to violate its schema.
        appends_eos = bool(getattr(contract, "appends_eos", True))
        problems = []
        if not comp0 or not comp1:
            problems.append("empty verdict completion sequence")
        if comp0 == comp1:
            problems.append("safe and unsafe completion sequences are identical")
        if safe_id == unsafe_id:
            problems.append("decision-position safe_id == unsafe_id")
        if appends_eos:
            # EOS-terminated contract: a COMPLETE verdict sequence terminates the schema.
            if eos is not None and (comp0[-1] != eos or comp1[-1] != eos):
                problems.append("appends_eos contract: verdict completion does not end in EOS")
            body0 = comp0[:-1] if (eos is not None and comp0[-1] == eos) else comp0
            body1 = comp1[:-1] if (eos is not None and comp1[-1] == eos) else comp1
        else:
            # native-schema guard: the supervised completion IS the top-level verdict sub-sequence
            # (no forced EOS); it must NOT terminate the schema early.
            if eos is not None and (comp0[-1] == eos or comp1[-1] == eos):
                problems.append("native-schema (appends_eos=False) contract forces a premature EOS")
            body0, body1 = comp0, comp1
        if not body0 or not body1:
            problems.append("verdict body is empty")
        status = "pass" if not problems else "fail"
        return _res("contract_tokens_distinct", status, {
            "contract": contract_name, "safe_id": safe_id, "unsafe_id": unsafe_id,
            "completion_ids": {"safe": comp0, "unsafe": comp1},
            "verdict_body_len": {"safe": len(body0), "unsafe": len(body1)},
            "eos_id": eos, "problems": problems,
        })
    except Exception as exc:  # NotImplementedError for stub guard contracts lands here
        return _fail("contract_tokens_distinct", exc)


def check_lora_targets_present(model, targets) -> dict:
    try:
        info = A.lora_targets_present(model, targets)
        status = "pass" if info["all_present"] else "fail"
        return _res("lora_targets_present", status, {
            "targets": list(targets), "present": info["present"],
            "all_present": info["all_present"]})
    except Exception as exc:
        return _fail("lora_targets_present", exc)


def check_adapter_disable_recovers(base_for_peft, ref_model, enc, recipe, tol,
                                   initial_kl_tol, device) -> dict:
    """Attach a fresh (step-zero, zero-init => identity) LoRA adapter and confirm disable_adapter()
    recovers the separately-loaded base within tol, with a finite initial KL ~ 0."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        lora = recipe["lora"]
        torch.manual_seed(0)
        peft = get_peft_model(base_for_peft, LoraConfig(
            r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=0.0,
            task_type="CAUSAL_LM", target_modules=list(lora["target_modules"])))
        peft.eval().to(device)
        ref_model.eval().to(device)
        out = A.adapter_disable_recovers(peft, ref_model, enc, tol=tol)
        finite_kl = bool(out["initial_kl"] == out["initial_kl"])  # not NaN
        kl_ok = finite_kl and abs(float(out["initial_kl"])) <= initial_kl_tol
        status = "pass" if (out["recovers"] and kl_ok) else "fail"
        return _res("adapter_disable_recovers", status, {
            "max_abs_logit_diff": out["max_abs_logit_diff"], "recovers": out["recovers"],
            "tol": tol, "initial_kl": out["initial_kl"], "initial_kl_tol": initial_kl_tol,
            "initial_kl_ok": kl_ok})
    except Exception as exc:
        return _fail("adapter_disable_recovers", exc)


def check_beta0_reproduces_sft(ckpt, contract_name, rows, recipe, device, work_dir) -> dict:
    """beta=0 through adapt() reproduces ordinary SFT. A one-step kl_sft(beta=0) update and a
    one-step sft update from the same seed and data order must yield a byte-identical adapter:
    identical post-step adapter bytes <=> identical 1-step loss and gradient. (The reference
    forward that only kl_sft performs happens AFTER the CE forward, so it cannot perturb the
    single supervised step.)"""
    try:
        one_step = dict(recipe)
        one_step["max_steps"] = 1
        sft_dir = os.path.join(work_dir, "beta0_sft")
        kl_dir = os.path.join(work_dir, "beta0_kl")
        m_sft = A.adapt(ckpt=ckpt, contract_name=contract_name, train_rows=rows,
                        method="sft", beta=0.0, seed=42, data_order_seed=42,
                        recipe=one_step, out_dir=sft_dir, device=device)
        m_kl = A.adapt(ckpt=ckpt, contract_name=contract_name, train_rows=rows,
                       method="kl_sft", beta=0.0, seed=42, data_order_seed=42,
                       recipe=one_step, out_dir=kl_dir, device=device)
        if m_sft["status"] != "completed" or m_kl["status"] != "completed":
            return _res("beta0_reproduces_sft", "fail", {
                "sft_status": m_sft["status"], "kl_status": m_kl["status"],
                "sft_reason": m_sft.get("failure_reason"),
                "kl_reason": m_kl.get("failure_reason")})
        sha_sft = m_sft.get("adapter_sha256")
        sha_kl = m_kl.get("adapter_sha256")
        equal = bool(sha_sft) and sha_sft == sha_kl
        return _res("beta0_reproduces_sft", "pass" if equal else "fail", {
            "adapter_sha256_sft": sha_sft, "adapter_sha256_klsft_beta0": sha_kl,
            "identical": equal, "max_steps": 1,
            "kl_beta": {"sft": m_sft["kl_beta"], "kl_sft": m_kl["kl_beta"]}})
    except Exception as exc:
        return _fail("beta0_reproduces_sft", exc)


def check_smoke_adapter_only_finite(ckpt, contract_name, rows, recipe, fixture_enc_builder,
                                    device, work_dir, tol) -> dict:
    """A small SFT smoke run changes ONLY the adapter (adapter-disabled logits still recover the
    released base) and produces finite, nonconstant unsafe margins across a fixture batch."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        smoke = dict(recipe)
        smoke["max_steps"] = int(recipe.get("max_steps", 2))
        out_dir = os.path.join(work_dir, "smoke_sft")
        meta = A.adapt(ckpt=ckpt, contract_name=contract_name, train_rows=rows,
                       method="sft", beta=0.0, seed=42, data_order_seed=42,
                       recipe=smoke, out_dir=out_dir, device=device)
        if meta["status"] != "completed":
            return _res("smoke_adapter_only_finite", "fail", {
                "status": meta["status"], "reason": meta.get("failure_reason")})
        adapter_dir = os.path.join(out_dir, "adapter")

        tok = A.load_study_tokenizer(ckpt)
        contract = A.get_contract(contract_name, tok)
        dec = contract.decision_ids(tok)
        enc = fixture_enc_builder(tok, contract, device)

        base = AutoModelForCausalLM.from_pretrained(
            ckpt["model_id"], revision=ckpt.get("model_revision"),
            dtype=C.torch_dtype_from_name(torch, str(ckpt.get("dtype", "float32"))),
            trust_remote_code=bool(ckpt.get("trust_remote_code", False))).eval().to(device)
        base_ref = AutoModelForCausalLM.from_pretrained(
            ckpt["model_id"], revision=ckpt.get("model_revision"),
            dtype=C.torch_dtype_from_name(torch, str(ckpt.get("dtype", "float32"))),
            trust_remote_code=bool(ckpt.get("trust_remote_code", False))).eval().to(device)
        peft = PeftModel.from_pretrained(base, adapter_dir).eval().to(device)

        with torch.no_grad():
            with peft.disable_adapter():
                disabled = peft(**enc).logits
            enabled = peft(**enc).logits
            base_logits = base_ref(**enc).logits
        base_unchanged = float((disabled - base_logits).abs().max())
        adapter_effect = float((enabled - disabled).abs().max())

        last = enc["attention_mask"].sum(1) - 1
        picked = enabled[torch.arange(enabled.shape[0]), last]
        margins = (picked[:, int(dec["unsafe_id"])] - picked[:, int(dec["safe_id"])]).float()
        finite = bool(torch.isfinite(margins).all())
        margin_std = float(margins.std().item()) if margins.numel() > 1 else 0.0
        nonconstant = margin_std > 0.0

        base_ok = base_unchanged <= tol
        status = "pass" if (base_ok and finite and nonconstant) else "fail"
        return _res("smoke_adapter_only_finite", status, {
            "smoke_rows": len(rows), "max_steps": smoke["max_steps"],
            "base_unchanged_max_abs_logit_diff": base_unchanged, "base_unchanged": base_ok,
            "adapter_effect_max_abs_logit_diff": adapter_effect,
            "margins_finite": finite, "margins_nonconstant": nonconstant,
            "margin_std": margin_std,
            "margins": [round(float(x), 4) for x in margins.tolist()], "tol": tol})
    except Exception as exc:
        return _fail("smoke_adapter_only_finite", exc)


def check_license_access_note(ckpt) -> dict:
    """Surface the registry's access/license note (proposal Sec 4.2/4.3). Informational: gated
    access must be secured manually in Phase 0; the preflight only records what the registry
    declares so a reviewer can confirm terms permit the experiment and a text-free score release."""
    access = ckpt.get("access")
    gated_terms = ("gated", "terms", "community")
    gated = bool(access) and any(t in str(access).lower() for t in gated_terms)
    note = (f"gated access declared ({access}); secure + snapshot terms before the lock"
            if gated else f"access={access!r} (ungated or general checkpoint)")
    return _res("license_access_note", "note", {
        "access": access, "gated": gated, "role": ckpt.get("role"),
        "native_output": ckpt.get("native_output"), "note": note})


# --------------------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------------------
def run_preflight(*, ckpt: dict, contract_name: str, recipe: dict,
                  fixture_rows: list[dict] | None = None, device: str = "cpu",
                  dtype: str = "float32", tol: float = DEFAULT_TOL,
                  initial_kl_tol: float = DEFAULT_INITIAL_KL_TOL,
                  smoke_rows: int = DEFAULT_SMOKE_ROWS,
                  include_training: bool = True, work_dir: str | None = None) -> dict:
    """Run the eligibility preflight for one checkpoint. Returns a report dict with a per-check
    list and an overall `eligible` bool (all REQUIRED checks pass). Fail-closed: unresolved model
    load skips model-dependent checks (recorded as skip), which cannot yield `eligible=True`."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    fixture_rows = fixture_rows or default_fixture_rows()
    max_len = int(recipe.get("max_length", 1024))
    # numeric-recovery checks need float32; the study dtype (bf16) blows the 1e-4 tolerance on CPU.
    load_dtype = dtype
    ckpt_cpu = copy.deepcopy(ckpt)
    ckpt_cpu["dtype"] = load_dtype

    report = {
        "starting_key": ckpt.get("starting_key"), "starting_type": ckpt.get("starting_type"),
        "model_id": ckpt.get("model_id"), "model_revision": ckpt.get("model_revision"),
        "tokenizer_revision": ckpt.get("tokenizer_revision"), "contract": contract_name,
        "device": device, "dtype": load_dtype, "tol": tol, "initial_kl_tol": initial_kl_tol,
        "finalization_status": "nonfinal", "created_utc": C.utcnow(),
        "checks": [], "eligible": False,
    }

    def add(r):
        report["checks"].append(r)

    # license/access is registry-only; always surface it.
    add(check_license_access_note(ckpt))

    owns_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="preflight_sta_")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # ---- revisions resolve (gates all model-dependent checks) ----
        tok = base_for_peft = ref_model = None
        try:
            tok = A.load_study_tokenizer(ckpt)
            td = C.torch_dtype_from_name(torch, load_dtype)
            base_for_peft = AutoModelForCausalLM.from_pretrained(
                ckpt["model_id"], revision=ckpt.get("model_revision"), dtype=td,
                trust_remote_code=bool(ckpt.get("trust_remote_code", False)))
            ref_model = AutoModelForCausalLM.from_pretrained(
                ckpt["model_id"], revision=ckpt.get("model_revision"), dtype=td,
                trust_remote_code=bool(ckpt.get("trust_remote_code", False)))
            add(_res("revisions_resolve", "pass", {
                "model_id": ckpt["model_id"], "model_revision": ckpt.get("model_revision"),
                "tokenizer_revision": ckpt.get("tokenizer_revision"), "dtype": load_dtype}))
            resolved = True
        except Exception as exc:
            add(_fail("revisions_resolve", exc))
            resolved = False

        if not resolved:
            for name in ("contract_tokens_distinct", "lora_targets_present",
                         "adapter_disable_recovers", "beta0_reproduces_sft",
                         "smoke_adapter_only_finite"):
                add(_res(name, "skip", {"reason": "model/tokenizer revisions did not resolve"}))
        else:
            # ---- contract tokens distinct/complete ----
            add(check_contract_tokens_distinct(contract_name, tok))

            # ---- lora targets present ----
            add(check_lora_targets_present(base_for_peft, recipe["lora"]["target_modules"]))

            # ---- adapter-disable recovers + initial KL ~ 0 (needs a rendered fixture batch) ----
            def build_enc(_tok, _contract, _device):
                prompts = [_contract.render(_tok, r["text"], max_len, reserved=8)[0]
                           for r in fixture_rows]
                enc = _tok(prompts, return_tensors="pt", padding=True, truncation=False,
                           add_special_tokens=False)
                return {k: v.to(_device) for k, v in enc.items()}

            try:
                contract = A.get_contract(contract_name, tok)
                enc = build_enc(tok, contract, device)
                add(check_adapter_disable_recovers(
                    base_for_peft, ref_model, enc, recipe, tol, initial_kl_tol, device))
            except Exception as exc:
                add(_fail("adapter_disable_recovers", exc))

            # free the shared reference models before the training sub-runs load their own.
            del base_for_peft, ref_model
            try:
                import gc
                gc.collect()
            except Exception:
                pass

            # ---- training checks (heavier; optional for a fast structural preflight) ----
            if include_training:
                add(check_beta0_reproduces_sft(
                    ckpt_cpu, contract_name, fixture_rows, recipe, device, work_dir))
                smoke_row_set = (fixture_rows if len(fixture_rows) >= smoke_rows
                                 else default_fixture_rows(smoke_rows))
                add(check_smoke_adapter_only_finite(
                    ckpt_cpu, contract_name, smoke_row_set, recipe, build_enc,
                    device, work_dir, tol))
            else:
                for name in ("beta0_reproduces_sft", "smoke_adapter_only_finite"):
                    add(_res(name, "skip", {"reason": "include_training=False"}))
    finally:
        if owns_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    required = {r["name"]: r for r in report["checks"] if r["required"]}
    report["eligible"] = bool(required) and all(
        required.get(name, {}).get("status") == "pass" for name in REQUIRED_CHECKS)
    report["summary"] = {
        "n_checks": len(report["checks"]),
        "passed": [r["name"] for r in report["checks"] if r["status"] == "pass"],
        "failed": [r["name"] for r in report["checks"] if r["status"] == "fail"],
        "skipped": [r["name"] for r in report["checks"] if r["status"] == "skip"],
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Starting-type adaptation eligibility preflight (Sec 4.3).")
    ap.add_argument("--config", default=DEFAULT_REGISTRY, help="registry YAML")
    ap.add_argument("--key", required=True, help="starting_key to preflight")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32",
                    help="load dtype for numeric checks (float32 recommended; bf16 blows tol)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--initial-kl-tol", type=float, default=DEFAULT_INITIAL_KL_TOL)
    ap.add_argument("--smoke-rows", type=int, default=DEFAULT_SMOKE_ROWS)
    ap.add_argument("--manifest", default=None, help="optional jsonl of fixture rows (text/label)")
    ap.add_argument("--fixture-limit", type=int, default=32)
    ap.add_argument("--skip-training", action="store_true",
                    help="run only the CPU-cheap structural checks (no adapt() sub-runs)")
    ap.add_argument("--out", default=None, help="write the report JSON here (default: stdout only)")
    args = ap.parse_args(argv)

    reg = A.load_registry(args.config)
    ckpts = reg["checkpoints"]
    if args.key not in ckpts:
        print(f"[preflight] unknown starting_key {args.key!r}; known: {sorted(ckpts)}",
              file=sys.stderr)
        return 2
    ckpt = ckpts[args.key]
    contract_name = ckpt.get("contract")
    fixture_rows = load_fixture_rows(args.manifest, args.fixture_limit)

    report = run_preflight(
        ckpt=ckpt, contract_name=contract_name, recipe=reg["recipe"],
        fixture_rows=fixture_rows, device=args.device, dtype=args.dtype, tol=args.tol,
        initial_kl_tol=args.initial_kl_tol, smoke_rows=args.smoke_rows,
        include_training=not args.skip_training)

    import json
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        C.write_json(args.out, report)
        print(f"[preflight] wrote {args.out}")
    print(text)
    print(f"[preflight] {args.key}: eligible={report['eligible']} "
          f"passed={len(report['summary']['passed'])} "
          f"failed={report['summary']['failed']} skipped={report['summary']['skipped']}")
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
