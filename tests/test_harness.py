import builtins
import sys
from types import SimpleNamespace

from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.evaluation.harness import evaluate


class _PerfectGuard:
    name = "perfect"

    def predict(self, text, *, surface=Surface.USER_PROMPT):
        unsafe = "unsafe" in text
        return Verdict(
            decision=Decision.UNSAFE if unsafe else Decision.SAFE,
            score=1.0 if unsafe else 0.0,
            surface=surface,
            latency_ms=1.0,
            model=self.name,
        )


def _samples():
    return [
        {"text": "safe example", "label": "safe"},
        {"text": "unsafe example", "label": "unsafe"},
    ]


def test_evaluate_works_without_mlflow(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    metrics = evaluate(_PerfectGuard(), _samples(), run_name="no-mlflow")
    assert metrics.f1 == 1.0
    assert metrics.accuracy == 1.0


def test_evaluate_logs_to_mlflow_if_present(monkeypatch):
    events = []

    class _Run:
        def __enter__(self):
            events.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit",))
            return False

    fake_mlflow = SimpleNamespace(
        start_run=lambda run_name=None: events.append(("start_run", run_name)) or _Run(),
        log_param=lambda key, value: events.append(("log_param", key, value)),
        log_metrics=lambda metrics: events.append(("log_metrics", metrics)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    metrics = evaluate(_PerfectGuard(), _samples(), run_name="with-mlflow")
    assert metrics.f1 == 1.0
    assert ("start_run", "with-mlflow") in events
    assert ("log_param", "guard", "perfect") in events
    assert any(event[0] == "log_metrics" and event[1]["f1"] == 1.0 for event in events)
