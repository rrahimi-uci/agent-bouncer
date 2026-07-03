"""Streaming progress for the Studio live console.

Long train/eval runs are otherwise silent — transformers' tqdm bar is carriage-return based
(dropped by the server's line reader) and the eval loop just runs. So both the training
callback and the scoring loop emit a compact, machine-readable ``PROGRESS`` marker via
:func:`progress_line`; the server parses it into a ``{"type": "progress"}`` event and the
console renders a live bar with step/%/loss/ETA — so you always see something moving.
"""

from __future__ import annotations

import time


def progress_line(step: int, total: int, *, phase: str = "train", label: str | None = None,
                  loss: float | None = None, rate: float | None = None,
                  eta: float | None = None, epoch: float | None = None) -> str:
    """Render a ``PROGRESS key=value ...`` marker (parsed by the server into a progress event).

    ``phase`` is ``train`` or ``test``; ``label`` names the current benchmark when testing.
    Optional fields (loss/rate/eta/epoch) are included only when known."""
    parts = [f"PROGRESS phase={phase}"]
    if label:
        parts.append(f"label={label}")
    parts.append(f"step={int(step)}")
    if total:
        parts.append(f"total={int(total)}")
        parts.append(f"pct={int(100 * step / max(1, total))}")
    if loss is not None:
        parts.append(f"loss={loss:.4f}")
    if rate is not None:
        parts.append(f"rate={rate:.2f}")
    if eta is not None:
        parts.append(f"eta={int(eta)}")
    if epoch is not None:
        parts.append(f"epoch={epoch:.2f}")
    return " ".join(parts)


class Throttle:
    """Rate-limit progress emission to at most one per ``min_interval`` seconds (plus forced
    emits, e.g. the final step). Keeps a run's console lively without flooding it."""

    def __init__(self, min_interval: float = 2.0) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def ready(self, now: float, *, force: bool = False) -> bool:
        if force or (now - self._last) >= self.min_interval:
            self._last = now
            return True
        return False


def progress_callback(min_interval: float = 2.0):  # pragma: no cover - needs transformers + a run
    """A ``TrainerCallback`` that streams throttled training progress (step/%/loss/rate/ETA) to
    the console. Works for the encoder ``Trainer`` and every TRL trainer (SFT/GRPO/DPO)."""
    from transformers import TrainerCallback

    class _ConsoleProgress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self._t0 = time.perf_counter()
            self._th = Throttle(min_interval)
            print("⚙️ Optimizer warming up — first step includes graph build / caching…", flush=True)

        def on_step_end(self, args, state, control, **kw):
            now = time.perf_counter()
            total = int(getattr(state, "max_steps", 0) or 0)
            step = int(getattr(state, "global_step", 0) or 0)
            if not self._th.ready(now, force=bool(total and step >= total)):
                return
            elapsed = now - self._t0
            rate = step / elapsed if elapsed > 0 else None
            eta = (total - step) / rate if (rate and total) else None
            loss = next((e["loss"] for e in reversed(getattr(state, "log_history", []) or [])
                         if "loss" in e), None)
            print(progress_line(step, total, phase="train", loss=loss, rate=rate, eta=eta,
                                epoch=getattr(state, "epoch", None)), flush=True)

    return _ConsoleProgress()
