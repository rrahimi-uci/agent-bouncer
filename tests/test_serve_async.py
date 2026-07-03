"""Cover the async run/stream internals (_run, _launch, run_events) with a fake subprocess."""

import asyncio

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

    async def wait(self):
        return self.returncode


def test_run_streams_parsed_events(monkeypatch):
    async def fake_exec(*a, **k):
        return _FakeProc([b"  [xstest] encoder-distilbert: P=0.5 R=0.5 F1=0.5 FPR=0.1 p50=5ms\n"])

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fake_exec)

    async def drive():
        rid = "rtest"
        api._RUNS[rid] = {"queue": asyncio.Queue(), "done": False, "proc": None}
        await api._run(rid, [["echo", "hi"]])
        q = api._RUNS[rid]["queue"]
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    events = asyncio.run(drive())
    types = [e["type"] for e in events]
    assert "step" in types and "result" in types and types[-1] == "done"


def test_run_reports_nonzero_exit(monkeypatch):
    async def fake_exec(*a, **k):
        return _FakeProc([b"boom\n"], code=1)

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fake_exec)

    async def drive():
        rid = "rerr"
        api._RUNS[rid] = {"queue": asyncio.Queue(), "done": False, "proc": None}
        await api._run(rid, [["x"]])
        q = api._RUNS[rid]["queue"]
        return [q.get_nowait() for _ in range(q.qsize())]

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
        api._RUNS[rid] = {"queue": asyncio.Queue(), "done": False, "proc": None}
        await api._run(rid, [["j1"], ["j2"], ["j3"]])
        q = api._RUNS[rid]["queue"]
        return [q.get_nowait() for _ in range(q.qsize())]

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
        api._RUNS[rid] = {"queue": asyncio.Queue(), "done": False, "proc": None}
        await api._run(rid, [["j1"], ["j2"]], stop_on_error=True)
        q = api._RUNS[rid]["queue"]
        return [q.get_nowait() for _ in range(q.qsize())]

    events = asyncio.run(drive())
    assert len([e for e in events if e["type"] == "step"]) == 1  # aborted after the first


def test_run_events_streams_sse():
    async def drive():
        rid = "rsse"
        q = asyncio.Queue()
        await q.put({"type": "log", "text": "hello"})
        await q.put({"type": "done"})
        api._RUNS[rid] = {"queue": q, "done": True, "proc": None}
        resp = await api.run_events(rid)
        return [chunk async for chunk in resp.body_iterator]

    chunks = asyncio.run(drive())
    assert any("hello" in c for c in chunks) and any("done" in c for c in chunks)
