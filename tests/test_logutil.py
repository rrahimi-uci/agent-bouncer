import logging

from agent_bouncer.config.logutil import DropSubstring, quiet_load_report


def test_drop_substring_keeps_and_drops():
    f = DropSubstring("LOAD REPORT")
    keep = logging.LogRecord("t", logging.WARNING, "", 0, "all good", None, None)
    drop = logging.LogRecord("t", logging.WARNING, "", 0, "MyModel LOAD REPORT from x", None, None)
    assert f.filter(keep) is True
    assert f.filter(drop) is False


def test_quiet_load_report_is_idempotent_and_filters():
    quiet_load_report()
    quiet_load_report()  # calling twice must not stack filters
    lg = logging.getLogger("transformers.modeling_utils")
    drops = [f for f in lg.filters if isinstance(f, DropSubstring)]
    assert len(drops) == 1
    rec = logging.LogRecord("t", logging.WARNING, "", 0, "X LOAD REPORT", None, None)
    assert drops[0].filter(rec) is False


def test_quiet_load_report_silences_pad_bos_eos_notice():
    quiet_load_report()
    quiet_load_report()  # idempotent for the trainer logger too
    lg = logging.getLogger("transformers.trainer_utils")
    drops = [f for f in lg.filters if isinstance(f, DropSubstring)]
    assert len(drops) == 1
    msg = ("The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and "
           "generation config. ... Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.")
    rec = logging.LogRecord("t", logging.WARNING, "", 0, msg, None, None)
    assert drops[0].filter(rec) is False                       # dropped
    keep = logging.LogRecord("t", logging.WARNING, "", 0, "real training warning", None, None)
    assert drops[0].filter(keep) is True                       # everything else survives


def test_pad_bos_eos_filter_is_model_agnostic():
    """The notice is a *fixed* transformers string — only the trailing token dict varies per
    model — so the single filter must drop it for every base model we train, not just Qwen."""
    quiet_load_report()
    lg = logging.getLogger("transformers.trainer_utils")
    drop = next(f for f in lg.filters if isinstance(f, DropSubstring))
    base = ("The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and "
            "generation config. The model config and generation config were aligned accordingly, "
            "being updated with the tokenizer's values. Updated tokens: {}.")
    per_model = {                                    # the exact dict transformers appends per model
        "qwen3":    "{'bos_token_id': None, 'pad_token_id': 151643}",
        "qwen3-1.7b": "{'bos_token_id': None, 'pad_token_id': 151643}",
        "smollm2":  "{'pad_token_id': 2}",          # Llama-style
        "gemma":    "{'pad_token_id': 0}",
        "deepseek": "{'bos_token_id': None, 'pad_token_id': 151643}",   # Qwen2-based distill
    }
    for tokens in per_model.values():
        rec = logging.LogRecord("transformers.trainer_utils", logging.WARNING, "", 0,
                                base.format(tokens), None, None)
        assert drop.filter(rec) is False            # suppressed for every model
