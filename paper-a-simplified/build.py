#!/usr/bin/env python3
"""Build the plain-language HTML and print PDF from README.md + GLOSSARY.md."""
from __future__ import annotations

import argparse
import pathlib
import posixpath
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote


HERE = pathlib.Path(__file__).resolve().parent
STEM = "the-benchmark-chooses-the-winner-simplified"
HTML = HERE / f"{STEM}.html"
PDF = HERE / f"{STEM}.pdf"
GITHUB_BLOB_ROOT = "https://github.com/rrahimi-uci/guard-ranking-fragility/blob/main/"
CHROME_CANDIDATES = (
    pathlib.Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    pathlib.Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
)


def run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def portable_print_html(html: str) -> str:
    """Replace repository-relative PDF links with stable GitHub URLs.

    Chrome otherwise serializes local links as absolute ``file:///Users/...``
    annotations, leaking the build path and producing a nonportable PDF.
    Fragment and already-absolute links are left alone.
    """
    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if target.startswith(("#", "http://", "https://", "mailto:")):
            return match.group(0)
        relative = posixpath.normpath(posixpath.join("paper-a-simplified", target))
        if relative == ".." or relative.startswith("../"):
            raise RuntimeError(f"print link escapes repository root: {target}")
        return f'href="{GITHUB_BLOB_ROOT}{quote(relative, safe="/:#?=&")}"'

    return re.sub(r'href="(?P<target>[^"]+)"', replace, html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    pandoc = shutil.which("pandoc")
    if not pandoc:
        parser.error("pandoc is required")

    readme = (HERE / "README.md").read_text()
    glossary = (HERE / "GLOSSARY.md").read_text()
    combined = (
        readme.replace("(GLOSSARY.md)", "(#glossary)")
        + "\n\n<div class=\"pagebreak\" id=\"glossary\"></div>\n\n"
        + glossary.replace("(README.md)", "(#the-benchmark-chooses-the-winner--plain-language-edition)")
    )
    with tempfile.TemporaryDirectory(prefix="paper-a-simplified-") as temporary:
        source = pathlib.Path(temporary) / "combined.md"
        source.write_text(combined)
        run([
            pandoc, str(source), "--from=gfm", "--to=html5", "--standalone",
            f"--template={HERE / 'template.html'}",
            "--metadata=title:The Benchmark Chooses the Winner — plain-language edition",
            f"--output={HTML}",
        ], cwd=HERE)

        if not args.html_only:
            chrome = next((str(path) for path in CHROME_CANDIDATES if path.is_file()), None)
            if not chrome:
                parser.error("Google Chrome or Chromium is required for PDF output")
            profile = pathlib.Path(temporary) / "chrome-profile"
            print_source = HERE / f".{STEM}-print.html"
            print_source.write_text(portable_print_html(HTML.read_text()))
            PDF.unlink(missing_ok=True)
            command = [
                chrome, "--headless", "--disable-gpu", "--disable-background-networking",
                "--no-first-run", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                f"--print-to-pdf={PDF}", print_source.as_uri(),
            ]
            try:
                run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except subprocess.TimeoutExpired:
                # Some Chrome builds keep the headless browser alive after successfully
                # printing. subprocess.run has terminated it; accept only a complete PDF.
                if not PDF.is_file() or PDF.stat().st_size < 10_000:
                    raise
            finally:
                print_source.unlink(missing_ok=True)

    if not HTML.is_file() or HTML.stat().st_size < 10_000:
        raise RuntimeError("plain-language HTML build is missing or unexpectedly small")
    if not args.html_only and (not PDF.is_file() or PDF.stat().st_size < 10_000):
        raise RuntimeError("plain-language PDF build is missing or unexpectedly small")
    if not args.html_only:
        pdf_bytes = PDF.read_bytes()
        if b"file:///" in pdf_bytes or b"/Users/" in pdf_bytes:
            raise RuntimeError("plain-language PDF contains a local build-path annotation")
    print(f"wrote {HTML.relative_to(HERE.parent)}")
    if not args.html_only:
        print(f"wrote {PDF.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
