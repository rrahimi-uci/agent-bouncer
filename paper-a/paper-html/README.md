# Paper — HTML edition

A clean, readable web edition of *"The Benchmark Chooses the Winner: Measuring
Fine-Tuning Specialization Across Safety-Guard Benchmarks,"* generated from the
LaTeX source.

## Open it

Double-click **`index.html`** (or drag it into a browser). No server needed;
the figures load from the sibling `figures/` folder. The top bar links to the
canonical PDF.

The folder is self-contained: zip or copy `paper-html/` and the reading edition,
its figures and offline math travel together.

## Features

- **Sticky section sidebar** (table of contents) with scroll-spy highlighting —
  jump to any section/subsection; a reading-progress bar tracks position.
- **Every table as a real HTML table** — numbered "Table N.", with the caption
  above and horizontal scrolling for wide tables.
- **Cross-references resolved** — every `\Cref`/`\ref` becomes a live link
  ("Table 7", "Figure 3", "§5.2") using the numbers from LaTeX's `.aux`.
- **Vector figures** — the plot figures are crisp SVG (converted from the PDF
  sources). TikZ schematic figures show their caption only.
- **Publication-quality math** — MathJax (SVG output), vendored locally under
  `vendor/mathjax/` so equations render offline with no CDN.
- Academic **serif** body type with a sans UI; **bibliography** via citeproc;
  links to the canonical PDF in the top bar.

## Regenerating

The edition is fully scripted, so it stays in sync with the paper:

```bash
python3 paper-a/paper-html/build.py
```

Requires `pandoc`, `tectonic` (for the `.aux` numbering), `pdftocairo`
(poppler, for PDF→SVG), and `curl` (first run only, to vendor MathJax). The
script rewrites `\begin{table*}`→`\begin{table}`
before pandoc (pandoc drops captions on the starred full-width float), reads the
`.aux` for authoritative numbers, runs pandoc with `template.html`, then
post-processes captions, cross-references, table wrappers, and figure images.

## Files

- `index.html` — the generated edition (open this).
- `template.html` — pandoc HTML template (layout, CSS, scroll-spy JS).
- `build.py` — the generator described above.
- `figures/` — SVG figures.
- `vendor/mathjax/` — vendored MathJax (offline SVG math).
- `explorer/` — an archived explorer for the earlier broad study. It is retained
  for history but is not linked from, or evidence for, focused Paper A.
