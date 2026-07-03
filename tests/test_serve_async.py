"""Cover the async run/stream internals (_run, _launch, run_events, history replay, stop/kill)
with a fake subprocess — no torch, no network."""

import asyncio
import os
import signal
import sys

import pytest

pytest.importorskip("fastapi")
from agent_bouncer.serving import api  # noqa: E402


class _FakeStdout:
    def __init__(self, lines):
        self._it = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeProc:
    def __init__(self, lines, code=0):
        self.stdout = _FakeStdout(lines)
        self.returncode = code
        self.pid = None            # _signal_group treats pid=None as a no-op

    async def wait(self):
        return self.returncode


def _mk_run(**kw):
    """A run record in the current shape (history + subscribers), overridable per test."""
    run = {"history": [], "subscribers": set(), "done": False, "proc": None, "stopped": False,
           "kind": "run", "created": 0.0}
    run.update(kw)
    return run


# ------------------------------------------------------------------- _run streaming
def test_run_streams_parsed_events(monkeypatch):
    async def fake_exec(*a, **k):
        return _FakeProc([b"  [xstest] encoder-distilbert: P=0.5 R=0.5 F1=0.5 FPR=0.1 p50=5ms\n"])

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fake_exec)

    async def drive():
        rid = "rtest"
        api._RUNS[rid] = _mk_run()
        await api._run(rid, [["echo", "hi"]])
        return api._RUNS[rid]["history"]

    events = asyncio.run(drive())
    types = [e["type"] for e in events]
    assert "step" in types and "result" in types and types[-1] == "done"
    assert all("_seq" in e for e in events)          # every event carries a sequence number


def test_run_reports_nonzero_exit(monkeypatch):
    async def fake_exec(*a, **k):
        return _FakeProc([b"boom\n"], code=1)

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fake_exec)

    async def drive():
        rid = "rerr"
        api._RUNS[rid] = _mk_run()
        await api._run(rid, [["x"]])
        return api._RUNS[rid]["history"]

    events = asyncio.run(drive())
    assert any(e["type"] == "error" for e in events)


def test_run_continues_past_a_failed_step(monkeypatch):
    # 3 jobs: ok, FAIL, ok — the failure must NOT abort the remaining jobs
    procs = iter([_FakeProc([b"a\n"], 0), _FakeProc([b"boom\n"], 1), _FakeProc([b"c\n"], 0)])

    async def fake_exec(*a, **k):
        return next(procs)

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fake_exec)

    async def drive():
        rid = "rmulti"
        api._RUNS[rid] = _mk_run()
        await api._run(rid, [["j1"], ["j2"], ["j3"]])
        return api._RUNS[rid]["history"]

    events = asyncio.run(drive())
    steps = [e for e in events if e["type"] == "step"]
    errs = [e for e in events if e["type"] == "error"]
    done = next(e for e in events if e["type"] == "done")
    assert len(steps) == 3                       # all three jobs ran despite the middle failure
    assert steps[0]["index"] == 1 and steps[2]["total"] == 3
    assert len(errs) == 1 and done["failures"] == 1 and done["total"] == 3


def test_run_stop_on_error_aborts(monkeypatch):
    procs = iter([_FakeProc([b"boom\n"], 1), _FakeProc([b"c\n"], 0)])

    async def fake_exec(*a, **k):
        return next(procs)

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fake_exec)

    async def drive():
        rid = "rstop"
        api._RUNS[rid] = _mk_run()
        await api._run(rid, [["j1"], ["j2"]], stop_on_error=True)
        return api._RUNS[rid]["history"]

    events = asyncio.run(drive())
    assert len([e for e in events if e["type"] == "step"]) == 1  # aborted after the first


# --------------------------------------------------------- history replay / reconnect
def test_run_events_replays_history_then_closes_when_done():
    async def drive():
        rid = "rsse"
        api._RUNS[rid] = _mk_run(done=True, history=[
            {"type": "log", "text": "hello", "_seq": 0}, {"type": "done", "_seq": 1}])
        resp = await api.run_events(rid)
        return [chunk async for chunk in resp.body_iterator]

    joined = "".join(asyncio.run(drive()))
    assert "hello" in joined and "done" in joined
    assert "_seq" not in joined          # internal sequence stripped from the wire


def test_run_events_replays_then_streams_live_without_dupes():
    # a reconnecting client replays existing history, then follows new events with no gaps/dupes
    async def drive():
        rid = "rlive"
        run = _mk_run(history=[{"type": "step", "index": 1, "total": 1, "_seq": 0}])
        api._RUNS[rid] = run
        it = (await api.run_events(rid)).body_iterator
        first = await it.__anext__()                 # replayed from history
        api._emit(run, {"type": "log", "text": "live-line"})
        api._emit(run, {"type": "done", "failures": 0, "total": 1})
        rest = "".join([chunk async for chunk in it])
        return first, rest

    first, rest = asyncio.run(drive())
    assert "step" in first and "live-line" in rest and "done" in rest


def test_run_events_unknown_run_404():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        asyncio.run(api.run_events("does-not-exist"))


def test_list_runs_endpoint():
    api._RUNS["lr1"] = _mk_run(kind="train", done=True, created=123.0,
                               history=[{"type": "step", "_seq": 0}])
    r = next(x for x in api.list_runs()["runs"] if x["run_id"] == "lr1")
    assert r["kind"] == "train" and r["done"] is True and r["events"] == 1


def test_prune_runs_drops_oldest_finished(monkeypatch):
    monkeypatch.setattr(api, "_RUNS", {})
    monkeypatch.setattr(api, "_MAX_RUNS", 2)
    api._RUNS["a"] = _mk_run(done=True, created=1.0)
    api._RUNS["b"] = _mk_run(done=True, created=2.0)
    api._RUNS["c"] = _mk_run(done=False, created=3.0)   # still running — must be kept
    api._prune_runs()
    assert "a" not in api._RUNS and "b" in api._RUNS and "c" in api._RUNS


# ----------------------------------------------------------------- stop / kill
def test_terminate_run_kills_a_real_subprocess():
    """End-to-end: a real child spawned in its own session is actually dead afterwards."""
    async def drive():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)", start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        killed = await api._terminate_run({"proc": proc}, grace=3.0)
        return killed, proc.returncode, proc.pid

    killed, rc, pid = asyncio.run(drive())
    assert killed is True and rc is not None and rc != 0
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)          # gone — signal 0 just probes existence


class _KillableProc:
    """A fake subprocess that 'dies' when its process group is signalled."""

    def __init__(self, pid=999):
        self.pid = pid
        self.returncode = None

    async def wait(self):
        self.returncode = -signal.SIGTERM
        return self.returncode


def test_terminate_run_kills_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(api.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(api.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    run = {"proc": _KillableProc(pid=321)}
    killed = asyncio.run(api._terminate_run(run, grace=1.0))
    assert killed is True and run["stopped"] is True
    assert (321, signal.SIGTERM) in calls          # whole group signalled, not just the pid


def test_terminate_run_escalates_to_sigkill(monkeypatch):
    calls = []
    monkeypatch.setattr(api.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(api.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    class Stubborn:
        def __init__(self):
            self.pid, self.returncode, self.n = 7, None, 0

        async def wait(self):
            self.n += 1
            if self.n == 1:
                await asyncio.sleep(0.05)   # ignores SIGTERM past the grace window
            self.returncode = -9
            return -9

    killed = asyncio.run(api._terminate_run({"proc": Stubborn()}, grace=0.01))
    assert killed is True
    assert (7, signal.SIGKILL) in calls          # escalated after the grace timeout


def test_terminate_run_noop_when_already_finished():
    run = {"proc": _FakeProc([b""], code=0)}     # returncode already set → nothing to kill
    assert asyncio.run(api._terminate_run(run)) is False
    assert run["stopped"] is True                # still flagged so no further jobs start


def test_run_stopped_before_start_emits_stopped_done():
    async def drive():
        rid = "rstop_pre"
        api._RUNS[rid] = _mk_run(stopped=True)
        await api._run(rid, [["j1"], ["j2"]])
        return api._RUNS[rid]["history"]

    events = asyncio.run(drive())
    done = next(e for e in events if e["type"] == "done")
    assert done["stopped"] is True and not any(e["type"] == "step" for e in events)


def test_run_stopped_during_reports_and_halts_batch(monkeypatch):
    async def fake_exec(*a, **k):
        api._RUNS["rstop_mid"]["stopped"] = True     # user pressed Stop mid-job
        return _FakeProc([b"working\n"], code=-15)

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fake_exec)

    async def drive():
        rid = "rstop_mid"
        api._RUNS[rid] = _mk_run()
        await api._run(rid, [["j1"], ["j2"]])
        return api._RUNS[rid]["history"]

    events = asyncio.run(drive())
    done = next(e for e in events if e["type"] == "done")
    assert done["stopped"] is True
    assert any("Stopped by user" in e.get("text", "") for e in events if e["type"] == "info")
    assert sum(e["type"] == "step" for e in events) == 1     # 2nd job never started


def test_stop_run_endpoint(monkeypatch):
    monkeypatch.setattr(api.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(api.os, "killpg", lambda pgid, sig: None)
    api._RUNS["known_stop"] = _mk_run(proc=_KillableProc(1))
    res = asyncio.run(api.stop_run("known_stop"))
    assert res["stopped"] is True and res["killed_process"] is True
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        asyncio.run(api.stop_run("no-such-run"))
