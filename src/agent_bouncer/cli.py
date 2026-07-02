"""`agent-bouncer` command-line interface.

`predict` and `eval` work today (via the reference KeywordGuard). `train` wires
into the SFT/GRPO entry points as they are implemented.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_bouncer import __version__
from agent_bouncer.core.guard import KeywordGuard
from agent_bouncer.core.schema import Surface


def _cmd_predict(args: argparse.Namespace) -> int:
    verdict = KeywordGuard().predict(args.text, surface=Surface(args.surface))
    print(json.dumps(verdict.model_dump(), indent=2, default=str))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from agent_bouncer.evaluation.harness import evaluate

    with open(args.data) as fh:
        samples = [json.loads(line) for line in fh if line.strip()]
    guard = KeywordGuard()  # TODO(--model): load a trained encoder/decoder guard
    metrics = evaluate(guard, samples, run_name=args.run_name)
    print(json.dumps(metrics.to_dict(), indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    if args.method == "sft":
        from agent_bouncer.training.sft import run_sft

        run_sft(args.config)
    elif args.method == "grpo":
        from agent_bouncer.training.grpo import run_grpo

        run_grpo(args.config)
    else:
        from agent_bouncer.training.dpo import run_dpo

        run_dpo(args.config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-bouncer",
        description="A tiny, fast safety bouncer for LLMs and agents.",
    )
    parser.add_argument("--version", action="version", version=f"agent-bouncer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Screen a single input")
    p_predict.add_argument("text")
    p_predict.add_argument("--surface", default="user_prompt", choices=[s.value for s in Surface])
    p_predict.set_defaults(func=_cmd_predict)

    p_eval = sub.add_parser("eval", help="Evaluate a guard over a JSONL dataset")
    p_eval.add_argument("data", help="JSONL with {'text': ..., 'label': 'safe'|'unsafe'} per line")
    p_eval.add_argument("--run-name", dest="run_name", default=None)
    p_eval.set_defaults(func=_cmd_eval)

    p_train = sub.add_parser("train", help="Train a guard")
    p_train.add_argument("method", choices=["sft", "grpo", "dpo"])
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
