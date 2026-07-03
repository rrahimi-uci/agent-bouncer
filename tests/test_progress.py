"""The pure progress helpers (progress_line marker + Throttle). The TrainerCallback itself
needs transformers + a live run, so it's excluded from coverage and not unit tested here."""

from agent_bouncer.training.progress import Throttle, progress_line


def test_progress_line_train_full_fields():
    line = progress_line(30, 60, phase="train", loss=0.4213, rate=2.714, eta=23, epoch=0.5)
    assert line.startswith("PROGRESS phase=train ")
    assert "step=30" in line and "total=60" in line and "pct=50" in line
    assert "loss=0.4213" in line and "rate=2.71" in line and "eta=23" in line and "epoch=0.50" in line


def test_progress_line_test_with_label_and_optional_fields():
    line = progress_line(5, 0, phase="test", label="beavertails")
    assert "phase=test" in line and "label=beavertails" in line and "step=5" in line
    # total=0 → no total/pct emitted; missing optionals omitted entirely
    assert "total=" not in line and "pct=" not in line
    assert "loss=" not in line and "rate=" not in line and "eta=" not in line


def test_throttle_rate_limits_but_allows_force():
    th = Throttle(2.0)
    assert th.ready(100.0) is True          # first call always ready
    assert th.ready(101.0) is False         # < 2s later → suppressed
    assert th.ready(101.5, force=True) is True   # forced (e.g. final step) → always emits
    assert th.ready(103.6) is True          # ≥ 2s since last emit → ready again
