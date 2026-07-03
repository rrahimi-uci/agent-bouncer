"""Tame third-party log noise so training / evaluation output stays readable."""

from __future__ import annotations

import logging

#: transformers loggers that emit the verbose model "LOAD REPORT" warning.
_TARGET_LOGGERS = ("transformers.modeling_utils", "transformers.integrations.peft")


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
    """Suppress transformers' verbose ``LOAD REPORT`` warning — the *benign* notice that a
    classifier head was newly initialized (expected when fine-tuning a base model into a
    guard) — while leaving every other warning intact. Idempotent; safe if transformers
    isn't installed (the loggers just never fire)."""
    for name in _TARGET_LOGGERS:
        lg = logging.getLogger(name)
        if not any(isinstance(f, DropSubstring) for f in lg.filters):
            lg.addFilter(DropSubstring("LOAD REPORT"))
