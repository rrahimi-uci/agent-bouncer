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
