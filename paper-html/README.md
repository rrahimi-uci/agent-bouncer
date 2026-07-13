# Paper — HTML edition

A clean, readable web edition of *"The Benchmark Chooses the Winner: A Fair
Evaluation of Small-LLM Safety Guards,"* generated from the LaTeX source.

## Open it

Double-click **`index.html`** (or drag it into a browser). No server needed;
the figures load from the sibling `figures/` folder.

## Features

- **Sticky section sidebar** (table of contents) with scroll-spy highlighting —
  jump to any section/subsection; a reading-progress bar tracks position.
- **All 21 tables as real HTML tables** — numbered "Table N.", with the caption
  above and horizontal scrolling for wide tables.
- **Cross-references resolved** — every `\Cref`/`\ref` becomes a live link
  ("Table 7", "Figure 3", "§5.2") using the numbers from LaTeX's `.aux`.
- **Vector figures** — the plot figures are crisp SVG (converted from the PDF
  sources). TikZ schematic figures show their caption only.
- **Math** via native MathML; **bibliography** via citeproc; links to the
  Benchmark Explorer and the canonical PDF in the top bar.

## Regenerating

The edition is fully scripted, so it stays in sync with the paper:

```
python3 paper-html/build.py
```

Requires `pandoc`, `tectonic` (for the `.aux` numbering), and `pdftocairo`
(poppler, for PDF→SVG). The script rewrites `\begin{table*}`→`\begin{table}`
before pandoc (pandoc drops captions on the starred full-width float), reads the
`.aux` for authoritative numbers, runs pandoc with `template.html`, then
post-processes captions, cross-references, table wrappers, and figure images.

## Files

- `index.html` — the generated edition (open this).
- `template.html` — pandoc HTML template (layout, CSS, scroll-spy JS).
- `build.py` — the generator described above.
- `figures/` — SVG figures.
