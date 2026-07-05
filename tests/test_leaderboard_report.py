"""Tests for the leaderboard PDF report builder (agent_bouncer.serving.leaderboard_report)."""

from agent_bouncer.serving import leaderboard_report as lr


def _g(p, r, f, auc, fpr, p50, p90):
    return dict(precision=p, recall=r, f1=f, roc_auc=auc, fpr_on_benign=fpr,
                latency_p50_ms=p50, latency_p90_ms=p90, throughput_per_s=1000 / max(p50, 1))


def _blob():
    return {
        "per_class": 50,
        "meta": {"xstest": {"axis": "over_refusal"}, "toxicchat": {"axis": "guardrail"}},
        "results": {
            "xstest": {
                "keyword-baseline": _g(.6, .5, .55, .6, .3, .2, .4),
                "encoder-modernbert-large": _g(.71, .70, .70, .79, .11, 44, 78),
                "openai-gpt-5.2-high": _g(.86, .85, .85, .87, .13, 4200, 6000),
                "ensemble-tuned": _g(.85, .86, .855, .90, .12, 120, 210),
            },
            "toxicchat": {
                "encoder-modernbert-large": _g(.72, .71, .71, .80, .10, 45, 80),
                "openai-gpt-5.2-high": _g(.88, .87, .87, .89, .11, 4300, 6100),
            },
        },
    }


def test_categorises_guards():
    assert lr._category("ensemble-tuned") == "Ensembles"
    assert lr._category("openai-gpt-5.2-high") == "GPT baselines"
    assert lr._category("encoder-modernbert-large") == "Small models"
    assert lr._category("keyword-baseline") == "Small models"


def test_label_prettifies_reasoning_tiers_and_ensembles():
    assert lr._label("openai-gpt-5.2-medium") == "GPT-5.2 medium"
    assert lr._label("ensemble-tuned") == "Ensemble · tuned"
    assert lr._label("encoder-modernbert-large") == "Encoder 395M"


def test_build_html_contains_all_rows_and_categories():
    html = lr.build_html(_blob(), sort="f1", generated="2026-07-03 12:00 UTC")
    assert "Agent Bouncer — Model Leaderboard" in html
    for cat in ("Small models", "GPT baselines", "Ensembles"):
        assert cat in html
    assert "GPT-5.2 high" in html and "Ensemble · tuned" in html
    # per-benchmark appendix lists both benchmarks
    assert "xstest" in html and "toxicchat" in html
    # best-in-column highlight is applied
    assert "num best" in html


def test_build_html_handles_empty_scoreboard():
    html = lr.build_html({"results": {}, "meta": {}}, sort="f1")
    assert "Agent Bouncer — Model Leaderboard" in html  # renders without crashing


def test_sort_falls_back_to_f1_on_unknown_key():
    # should not raise on an unknown sort key
    html = lr.build_html(_blob(), sort="not-a-metric")
    assert "Model Leaderboard" in html


def test_macro_average_yields_none_for_missing_metric():
    # a metric absent everywhere must be None (→ "—"), not zero-filled (→ fake 0.000 "best")
    results = {"b1": {"g": {"precision": .7, "recall": .6, "f1": .65, "fpr_on_benign": .1,
                            "latency_p50_ms": 5, "latency_p90_ms": 9, "throughput_per_s": 100}}}
    m = lr._macro_average(results)
    assert m["g"]["roc_auc"] is None and m["g"]["f1"] == .65
    html = lr.build_html({"results": results, "meta": {}})
    assert "—" in html and "num best" in html   # renders em-dash, still highlights real bests
