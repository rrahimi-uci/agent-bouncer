from agent_bouncer.eval.report import generate_model_card, render_results_table


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
