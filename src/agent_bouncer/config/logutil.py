"""Tame third-party log noise so training / evaluation output stays readable."""

from __future__ import annotations

import logging

#: Benign, verbose transformers notices to drop, mapped to the logger that emits each.
#: These are *informational* (transformers telling you it already reconciled something) —
#: not errors — so hiding them keeps the console clean without masking real warnings.
_NOISE: dict[str, tuple[str, ...]] = {
    # A classifier head was newly initialized — expected when fine-tuning a base model into a guard.
    "transformers.modeling_utils": ("LOAD REPORT",),
    "transformers.integrations.peft": ("LOAD REPORT",),
    # The tokenizer's PAD/BOS/EOS differed from the model config, so transformers aligned them
    # (e.g. Qwen uses <|endoftext|> as the pad token and no BOS). Purely a courtesy heads-up.
    "transformers.trainer_utils": ("new PAD/BOS/EOS tokens",),
}


class DropSubstring(logging.Filter):
    """A logging filter that drops any record whose message contains ``needle``."""

    def __init__(self, needle: str) -> None:
        super().__init__()
        self.needle = needle

    def filter(self, record: logging.LogRecord) -> bool:  # True = keep
        try:
            return self.needle not in record.getMessage()
        except Exception:  # noqa: BLE001 - logging must never raise
            return True


def quiet_load_report() -> None:
    """Suppress a couple of *benign* transformers notices — the ``LOAD REPORT`` warning (a
    newly-initialized classifier head) and the PAD/BOS/EOS token-alignment heads-up — while
    leaving every other warning intact. Idempotent (won't stack filters); safe if transformers
    isn't installed (the loggers just never fire)."""
    for name, needles in _NOISE.items():
        lg = logging.getLogger(name)
        existing = {f.needle for f in lg.filters if isinstance(f, DropSubstring)}
        for needle in needles:
            if needle not in existing:
                lg.addFilter(DropSubstring(needle))
