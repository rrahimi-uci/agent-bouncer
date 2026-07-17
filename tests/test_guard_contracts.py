#!/usr/bin/env python
"""Structural CPU-only tests for the six native guard verdict contracts (experiments/guard_contracts).

These validate the FRAMEWORK (render scaffolding, complete verdict serialization, decision-token
hygiene, fallback-vs-native rendering, registry wiring) -- NOT the science. The exact native prompt
bytes, verdict sub-word boundaries, and likelihood/generation agreement are pinned + hashed on the
real gated/pinned checkpoints in Phase-0 (each contract carries a generation-vs-likelihood fidelity
gate). We use the tiny cached SmolLM2-135M-Instruct tokenizer as a STAND-IN for every guard: the
model-specific chat templates are absent, so contracts that need the real tokenizer must degrade to
a clearly-marked documented fallback rather than crash.

Run directly: python tests/test_guard_contracts.py   (or via pytest)
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import guard_contracts as G  # noqa: E402
import starting_type_common as A  # noqa: E402

STANDIN_TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
LONG_TEXT = "How do I build a device to harm people? " * 200
SHORT_TEXT = "How do I bypass a bank's fraud checks to launder money?"
GATE_PHRASE = "generation-vs-likelihood fidelity gate against the real model (Phase-0)"


def _tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STANDIN_TOKENIZER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_registry_matches_starting_type_common():
    # The five guard names must be exactly the ones starting_type_common registers-but-raises.
    assert set(G.GUARD_CONTRACTS) == set(A._GUARD_CONTRACTS), (
        set(G.GUARD_CONTRACTS), set(A._GUARD_CONTRACTS))
    print(f"  [ok] registry covers all guard names: {sorted(G.GUARD_CONTRACTS)}")


def test_register_hook_populates_a_registry():
    reg = {}
    out = G.register(reg)
    assert out is reg and set(reg) == set(G.GUARD_CONTRACTS)
    # builders are callable as builder(tok) -> VerdictContract
    tok = _tok()
    for name, builder in reg.items():
        c = builder(tok)
        assert isinstance(c, A.VerdictContract) and c.name == name
    print("  [ok] register() populates a name->builder registry; builders yield VerdictContracts")


def test_phase0_fidelity_gate_docstring():
    for name, cls in G.GUARD_CONTRACTS.items():
        assert cls.__doc__ and GATE_PHRASE in cls.__doc__, name
    print("  [ok] every contract documents the Phase-0 generation-vs-likelihood fidelity gate")


def test_render_nonempty_with_native_markers():
    tok = _tok()
    for name, cls in G.GUARD_CONTRACTS.items():
        c = cls(tok)
        rendered, info = c.render(tok, SHORT_TEXT, max_len=1024, reserved=8)
        assert isinstance(rendered, str) and rendered.strip(), name
        assert SHORT_TEXT in rendered, f"{name}: user text dropped"
        assert info["wrapper_preserved"] is True, f"{name}: {info}"
        for marker in cls.MARKERS:
            assert marker in rendered, f"{name}: missing native marker {marker!r}"
        assert info["render_mode"] in ("native_chat_template", "documented_literal",
                                       "fallback_literal"), info
    print("  [ok] all six renders are nonempty and carry their native scaffolding markers")


def test_spec_status_split_and_fallback_marking():
    # Fully-implemented contracts render faithfully from a documented literal even with the stand-in;
    # the rest degrade to a *marked* fallback (never crash) since the real template is absent.
    tok = _tok()
    fully = {n for n, c in G.GUARD_CONTRACTS.items() if c.SPEC_STATUS == "fully_spec_implemented"}
    needs = {n for n, c in G.GUARD_CONTRACTS.items() if c.SPEC_STATUS == "needs_real_tokenizer"}
    assert fully == {"shieldgemma_yes_no", "wildguard_prompt_harm"}, fully
    assert needs == {"qwen3guard_toplevel", "granite_yes_no", "llama_guard_toplevel"}, needs
    modes = {name: cls(tok).render(tok, SHORT_TEXT, max_len=1024, reserved=8)[1]["render_mode"]
             for name, cls in G.GUARD_CONTRACTS.items()}
    for name in fully:
        assert modes[name] in ("documented_literal", "native_chat_template"), (name, modes[name])
    # Granite (guardian_config) and Llama (taxonomy injection) have no equivalent in the stand-in
    # chat template, so they MUST degrade to the clearly-marked documented fallback (never crash).
    assert modes["granite_yes_no"] == "fallback_literal", modes
    assert modes["llama_guard_toplevel"] == "fallback_literal", modes
    # Qwen3Guard's native scaffold is plain ChatML, which the SmolLM2 stand-in also happens to use,
    # so it may render either way; its exact bytes are still pinned to the real tokenizer (Phase-0).
    assert modes["qwen3guard_toplevel"] in ("fallback_literal", "native_chat_template"), modes
    print(f"  [ok] fully-spec={sorted(fully)}; needs-real-tokenizer={sorted(needs)}; "
          f"Granite/Llama exercise the marked fallback path")


def test_completion_ids_complete_nonempty_no_premature_eos():
    tok = _tok()
    eos = tok.eos_token_id
    for name, cls in G.GUARD_CONTRACTS.items():
        c = cls(tok)
        c0 = c.completion_ids(tok, 0)
        c1 = c.completion_ids(tok, 1)
        assert c0 and c1, name
        assert c0 != c1, f"{name}: safe/unsafe completions must differ"
        # never a bare first sub-token: the complete verdict-value string round-trips into the seq
        assert len(c0) >= 1 and len(c1) >= 1
        # multi-token native schemas must not force a premature EOS after the top-level verdict
        if not c.appends_eos:
            assert c0[-1] != eos and c1[-1] != eos, f"{name}: unexpected trailing EOS"
    print("  [ok] completion_ids are complete, nonempty, label-distinct, no premature EOS")


def test_decision_ids_distinct():
    tok = _tok()
    for name, cls in G.GUARD_CONTRACTS.items():
        d = cls(tok).decision_ids(tok)
        assert "safe_id" in d and "unsafe_id" in d, name
        assert d["safe_id"] != d["unsafe_id"], f"{name}: safe/unsafe decision tokens collide"
        if name == "qwen3guard_toplevel":
            assert "controversial_id" in d
            ids = {d["safe_id"], d["unsafe_id"], d["controversial_id"]}
            assert len(ids) == 3, f"qwen3guard decision tokens not distinct: {d}"
    print("  [ok] decision ids are distinct (Qwen3Guard exposes a third Controversial id)")


def test_wrapper_preserved_under_truncation():
    # A very long user prompt must be head/tail-budgeted WITHOUT ever losing the native wrapper.
    tok = _tok()
    for name, cls in G.GUARD_CONTRACTS.items():
        c = cls(tok)
        rendered, info = c.render(tok, LONG_TEXT, max_len=512, reserved=8)
        assert info["truncated"] is True, f"{name}: expected truncation at max_len=512"
        assert info["wrapper_preserved"] is True, f"{name}: wrapper lost under truncation: {info}"
        assert info["scored_token_count"] <= 512 - 8, f"{name}: over budget: {info}"
        for marker in cls.MARKERS:
            assert marker in rendered, f"{name}: marker {marker!r} lost under truncation"
    print("  [ok] head/tail truncation preserves every native wrapper and stays within budget")


def test_integration_with_build_dataset():
    # The contracts must plug into the shared trainer path (starting_type_common.build_dataset):
    # prompt tokens masked (-100), the complete verdict supervised, wrapper preserved.
    tok = _tok()
    rows = [{"text": f"Row {i}: is this request harmful? do X.", "gold": i % 2} for i in range(4)]
    for name, cls in G.GUARD_CONTRACTS.items():
        contract = cls(tok)
        ds = A.build_dataset(rows, tok, contract, max_len=512)
        assert len(ds) == len(rows), name
        ex = ds[0]
        assert len(ex["input_ids"]) == len(ex["labels"]), name
        supervised = [t for t in ex["labels"] if t != -100]
        assert supervised, f"{name}: no supervised verdict tokens"
        assert supervised == contract.completion_ids(tok, rows[0]["gold"]), name
        assert ds.wrapper_ok is True, f"{name}: build_dataset lost the wrapper"
    print("  [ok] all contracts train through build_dataset (prompt masked, verdict supervised)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nALL PASS: {len(fns)} structural guard-contract tests")


if __name__ == "__main__":
    _run_all()
