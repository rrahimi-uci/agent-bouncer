#!/usr/bin/env python
"""Phase 7: turn demo results into a Markdown table + a model card.

Reads outputs/demo_results.json (written by demo_train_eval.py) and writes
outputs/RESULTS.md and outputs/MODEL_CARD.md.
"""

from __future__ import annotations

import json

from agent_bouncer.evaluation.report import generate_model_card, render_results_table


def main() -> None:
    with open("outputs/demo_results.json") as fh:
        res = json.load(fh)

    rows = [
        {"guard": "keyword-baseline", "params": "0", **res["keyword"]},
        {"guard": "agent-bouncer-encoder", "params": "66M", **res["encoder"]},
    ]
    table = render_results_table(rows)
    print(table)

    with open("outputs/RESULTS.md", "w") as fh:
        fh.write("# Agent Bouncer — demo results\n\n")
        fh.write(f"Held-out test set: {res.get('test_n', '?')} BeaverTails prompts.\n\n")
        fh.write(table + "\n")

    card = generate_model_card(
        "agent-bouncer-encoder",
        "distilbert-base-uncased",
        res["encoder"],
        datasets=["PKU-Alignment/BeaverTails"],
    )
    with open("outputs/MODEL_CARD.md", "w") as fh:
        fh.write(card)

    print("\nwrote outputs/RESULTS.md and outputs/MODEL_CARD.md")


if __name__ == "__main__":
    main()
