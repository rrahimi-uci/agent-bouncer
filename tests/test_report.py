import pytest

from agent_bouncer.evaluation.report import (
    generate_model_card,
    macro_average,
    render_benchmark_report,
    render_results_table,
)


def test_results_table_has_header_and_rows():
    rows = [
        {"guard": "keyword-baseline", "params": "0", "f1": 0.5, "recall": 0.4,
         "fpr_on_benign": 0.02, "latency_p50_ms": 0.01},
        {"guard": "encoder", "params": "66M", "f1": 0.91, "recall": 0.9,
         "fpr_on_benign": 0.05, "latency_p50_ms": 8.0},
    ]
    table = render_results_table(rows)
    lines = table.splitlines()
    assert lines[0].startswith("| Guard")
    assert "---" in lines[1]
    assert "keyword-baseline" in table
    assert "0.910" in table  # floats formatted to 3dp


_RESULTS = {
    "beavertails": {
        "encoder": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "fpr_on_benign": 0.2, "latency_p50_ms": 7.0},
        "gpt": {"precision": 0.6, "recall": 0.9, "f1": 0.72, "fpr_on_benign": 0.4, "latency_p50_ms": 700.0},
    },
    "prompt_injections": {
        "encoder": {"precision": 0.5, "recall": 0.4, "f1": 0.45, "fpr_on_benign": 0.1, "latency_p50_ms": 7.0},
        # gpt skipped on this one -> should not be penalised in its average
    },
}
_META = {
    "beavertails": {"axis": "guardrail", "description": "harm", "n_safe": 10, "n_unsafe": 10},
    "prompt_injections": {"axis": "red_team", "description": "injection", "n_safe": 5, "n_unsafe": 5},
}


def test_macro_average_only_counts_scored_benchmarks():
    avg = macro_average(_RESULTS)
    # encoder scored on both benchmarks -> mean of the two F1s
    assert avg["encoder"]["f1"] == pytest.approx((0.65 + 0.45) / 2)
    # gpt scored on only one -> its single F1, not diluted by a zero
    assert avg["gpt"]["f1"] == pytest.approx(0.72)


def test_render_benchmark_report_groups_by_axis_and_lists_gated():
    report = render_benchmark_report(
        _RESULTS, _META, {"encoder": "66M", "gpt": "api"},
        guard_order=["encoder", "gpt"], gated={"wildguardmix": "allenai/wildguardmix"},
    )
    assert "Guardrail" in report and "Red-teaming" in report
    assert "`beavertails`" in report and "`prompt_injections`" in report
    assert "Macro-average" in report
    assert "wildguardmix" in report  # gated benchmarks are disclosed, not hidden


def test_model_card_has_frontmatter_and_sections():
    card = generate_model_card(
        "agent-bouncer-encoder",
        "distilbert-base-uncased",
        {"f1": 0.91, "fpr_on_benign": 0.05},
        datasets=["PKU-Alignment/BeaverTails"],
    )
    assert card.startswith("---")
    assert "license: apache-2.0" in card
    assert "base_model: distilbert-base-uncased" in card
    assert "## Evaluation" in card
    assert "PKU-Alignment/BeaverTails" in card
