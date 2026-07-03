"""Render the Workbench leaderboard as a self-contained HTML report and convert it to
PDF with headless Chrome (no extra Python deps).

The leaderboard is the macro-average (across benchmarks) of every guard scored in
``outputs/benchmark_results.json`` — the same numbers the dashboard's Leaderboard tab
shows — grouped into Small models / GPT baselines / Ensembles.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import tempfile

from agent_bouncer.evaluation.report import macro_average

# Columns: (metric key, header, "hi"=higher-better | "lo"=lower-better, decimals).
_COLUMNS = [
    ("precision", "Precision", "hi", 3),
    ("recall", "Recall", "hi", 3),
    ("f1", "F1", "hi", 3),
    ("roc_auc", "AUC", "hi", 3),
    ("fpr_on_benign", "FPR@benign", "lo", 3),
    ("latency_p50_ms", "p50 ms", "lo", 0),
    ("latency_p90_ms", "p90 ms", "lo", 0),
]
_SORTABLE = {k for k, *_ in _COLUMNS}
_CAT_ORDER = ["Small models", "GPT baselines", "Ensembles"]

_PARAMS = {
    "keyword-baseline": "0", "encoder-distilbert": "66M", "encoder-modernbert-large": "395M",
    "modernbert-large": "395M", "decoder-sft-0.6B": "0.6B", "decoder-grpo-0.6B": "0.6B",
    "decoder-sft-1.7B": "1.7B",
}
_LABELS = {
    "keyword-baseline": "Keyword", "encoder-distilbert": "Encoder 66M",
    "encoder-modernbert-large": "Encoder 395M",
    "decoder-sft-0.6B": "Decoder-SFT 0.6B", "decoder-sft-1.7B": "Decoder-SFT 1.7B",
    "decoder-grpo-0.6B": "Decoder-GRPO", "openai-moderation": "OpenAI-Mod",
    "openai-gpt-4o-mini": "GPT-4o-mini",
}


def _category(guard: str) -> str:
    if guard.startswith("ensemble-"):
        return "Ensembles"
    if guard.startswith("openai-"):
        return "GPT baselines"
    return "Small models"


def _label(guard: str) -> str:
    if guard in _LABELS:
        return _LABELS[guard]
    import re
    m = re.match(r"^openai-gpt-([\d.]+)-(low|medium|high)$", guard)
    if m:
        return f"GPT-{m.group(1)} {m.group(2)}"
    if guard.startswith("ensemble-"):
        return "Ensemble · " + guard[len("ensemble-"):]
    return guard


def _params(guard: str) -> str:
    if guard in _PARAMS:
        return _PARAMS[guard]
    if guard.startswith("openai-"):
        return "api"
    import re
    m = re.search(r"([\d.]+[BM])", guard)
    return m.group(1) if m else "—"


def _fmt(value: float | None, decimals: int) -> str:
    if value is None:
        return "—"
    if decimals == 0:
        return f"{value:.1f}" if value < 10 else f"{round(value)}"
    return f"{value:.{decimals}f}"


def build_html(blob: dict, sort: str = "f1", *, generated: str = "") -> str:
    """Build the standalone HTML leaderboard report from a benchmark_results blob."""
    results = blob.get("results", {})
    per_class = blob.get("per_class")
    summary = macro_average(results)
    benches = sorted(results.keys())

    sort = sort if sort in _SORTABLE else "f1"
    lo = next((c[2] == "lo" for c in _COLUMNS if c[0] == sort), False)

    # best value per column (across ALL guards) for highlighting
    best: dict[str, float] = {}
    for key, _, hl, _dp in _COLUMNS:
        vals = [m[key] for m in summary.values() if m.get(key) is not None]
        if vals:
            best[key] = min(vals) if hl == "lo" else max(vals)

    head = "".join(
        f'<th class="{"lo" if hl == "lo" else "hi"}">{html.escape(title)} '
        f'{"↓" if hl == "lo" else "↑"}</th>'
        for _k, title, hl, _dp in _COLUMNS
    )
    rows_html: list[str] = []
    for cat in _CAT_ORDER:
        guards = [g for g in summary if _category(g) == cat]
        if not guards:
            continue
        guards.sort(key=lambda g: summary[g].get(sort, 0.0) or 0.0, reverse=not lo)
        rows_html.append(
            f'<tr class="cat"><td colspan="{len(_COLUMNS) + 2}">{html.escape(cat)} · {len(guards)}</td></tr>'
        )
        for g in guards:
            m = summary[g]
            cells = []
            for key, _t, _hl, dp in _COLUMNS:
                v = m.get(key)
                is_best = v is not None and key in best and abs(v - best[key]) < 1e-9
                cls = "num best" if is_best else "num"
                cells.append(f'<td class="{cls}">{_fmt(v, dp)}</td>')
            rows_html.append(
                f'<tr><td class="name">{html.escape(_label(g))}</td>'
                f'<td class="num muted">{html.escape(_params(g))}</td>{"".join(cells)}</tr>'
            )

    # per-benchmark F1 appendix (guard rows × benchmark cols)
    bench_head = "".join(f"<th>{html.escape(b)}</th>" for b in benches)
    bench_rows = []
    for g in sorted(summary, key=lambda g: (_CAT_ORDER.index(_category(g)), -(summary[g].get("f1") or 0))):
        tds = "".join(
            f'<td class="num">{_fmt((results[b].get(g) or {}).get("f1"), 3)}</td>' for b in benches
        )
        bench_rows.append(f'<tr><td class="name">{html.escape(_label(g))}</td>{tds}</tr>')

    subtitle = (
        f"Macro-averaged over {len(benches)} benchmark(s)"
        + (f", ≤{per_class}/class" if per_class else "")
        + ". Positive class = <code>unsafe</code>."
    )
    gen_line = f"<span>Generated {html.escape(generated)}</span>" if generated else ""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>Agent Bouncer — Model Leaderboard</title>
<style>
  @page {{ size: A4 landscape; margin: 16mm 14mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #12151c; margin: 0; font-size: 12px; }}
  h1 {{ font-size: 21px; margin: 0 0 2px; letter-spacing: -.3px; }}
  .meta {{ color: #5c6b88; font-size: 12px; margin: 0 0 18px; display: flex; gap: 16px; flex-wrap: wrap; }}
  h2 {{ font-size: 14px; margin: 22px 0 8px; color: #1b2338; }}
  code {{ background: #eef1f7; padding: 1px 5px; border-radius: 4px; font-size: 11px; }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 6px 9px; border-bottom: 1px solid #e5e9f2; white-space: nowrap; }}
  th {{ font-size: 10px; text-transform: uppercase; letter-spacing: .05em; color: #7a879f;
    border-bottom: 1.5px solid #cfd6e4; }}
  th:first-child, td:first-child {{ text-align: left; }}
  td.name {{ font-weight: 600; }}
  td.num {{ font-feature-settings: "tnum"; }}
  td.muted {{ color: #8a97b0; }}
  td.best {{ color: #0a8f5b; font-weight: 700; }}
  tr.cat td {{ background: #f4f6fb; color: #6b7896; font-size: 9.5px; text-transform: uppercase;
    letter-spacing: .07em; font-weight: 700; padding-top: 9px; }}
  .note {{ color: #7a879f; font-size: 10.5px; margin-top: 10px; line-height: 1.5; }}
  .foot {{ margin-top: 20px; color: #9aa6bd; font-size: 10px;
    border-top: 1px solid #e5e9f2; padding-top: 8px; }}
</style></head><body>
  <h1>Agent Bouncer — Model Leaderboard</h1>
  <div class="meta"><span>{subtitle}</span>{gen_line}</div>
  <table><thead><tr><th>Model</th><th class="hi">Params</th>{head}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody></table>
  <p class="note">Sorted by <b>{html.escape(sort)}</b>.
    Best value per column in <b style="color:#0a8f5b">green</b>.
    AUC for hard-decision guards (GPT, keyword) is the single-point estimate (recall+1−FPR)/2;
    the encoder &amp; threshold-swept ensembles use a true swept AUC.</p>
  <h2>Per-benchmark F1</h2>
  <table><thead><tr><th>Model</th>{bench_head}</tr></thead><tbody>{"".join(bench_rows)}</tbody></table>
  <div class="foot">Agent Bouncer — SLM guardrails · all guards scored through one harness.</div>
</body></html>"""


def _find_chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    for path in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ):
        if os.path.exists(path):
            return path
    return None


def render_pdf(html_str: str, out_path: str, *, timeout: float = 45.0) -> None:
    """Convert an HTML string to a PDF at ``out_path`` using headless Chrome.

    Headless Chrome writes the PDF within a couple of seconds but does not always exit
    cleanly on macOS, so we poll for the output file and then tear the process down —
    rather than blocking on the process exiting.

    Raises ``RuntimeError`` if no Chrome/Chromium is available or no PDF is produced.
    """
    import signal
    import time

    chrome = _find_chrome()
    if not chrome:
        raise RuntimeError(
            "No Chrome/Chromium found for PDF rendering. Install Google Chrome, or "
            "export the leaderboard from the browser's print dialog."
        )
    tmpdir = tempfile.mkdtemp(prefix="ab-report-")
    html_path = os.path.join(tmpdir, "report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html_str)
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
        "--no-default-browser-check", "--disable-crash-reporter", "--disable-breakpad",
        "--disable-dev-shm-usage", "--disable-extensions",
        f"--user-data-dir={os.path.join(tmpdir, 'profile')}", "--no-pdf-header-footer",
        "--virtual-time-budget=8000", f"--print-to-pdf={out_path}", f"file://{html_path}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                time.sleep(0.3)  # let the final flush complete
                return
            if proc.poll() is not None:  # Chrome exited (success writes the file above)
                break
            time.sleep(0.25)
        raise RuntimeError("PDF rendering timed out or produced no output")
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)
