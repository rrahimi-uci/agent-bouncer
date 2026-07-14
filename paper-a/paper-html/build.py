#!/usr/bin/env python3
"""Generate the HTML edition of the paper from the LaTeX source.

Pipeline (all reproducible; rerun after the paper's numbers change):
  1. figures: vector PDF -> SVG (pdftocairo), crisp + offline.
  2. numbers: compile once with tectonic to read the .aux -> authoritative
     label -> number map (Table 5, Figure 3, section 5.2, ...).
  3. body: pandoc LaTeX -> HTML5 with our template (sidebar TOC, MathJax math,
     citeproc bibliography). We rewrite  table* -> table  first because pandoc
     drops captions/labels on the starred (full-width) float.
  4. post-process the emitted HTML:
       - number every table caption ("Table N. ...") and figure caption,
       - resolve every \\Cref/\\ref cross-reference (incl. multi-label and
         section refs) into a real linked "Table N" / "Figure N" / "§N",
       - wrap wide tables in a horizontal-scroll container,
       - point <img> at the SVG figures.
     The sidebar TOC is protected from ref-rewriting.

Run from the repository root:  python3 paper-a/paper-html/build.py
"""
import os, re, shutil, subprocess
from itertools import groupby

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
TEX = "benchmark_chooses_the_winner.tex"
INPUTS = [
    "results_macros_gen.tex",
    "tab_primary_gen.tex",
    "tab_sensitivity_gen.tex",
    "tab_seed_values_gen.tex",
]  # \input'd generated values/tabulars
OUT = os.path.join(HERE, "index.html")
FIGDIR = os.path.join(HERE, "figures")
TMP = "/tmp/guard-ranking-fragility-paper-html"
AUXDIR = "/tmp/guard-ranking-fragility-paper-aux"

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)

# ---------------------------------------------------------------- 0. vendor MathJax (offline SVG math)
MJ = os.path.join(HERE, "vendor", "mathjax", "tex-svg.js")
if not os.path.exists(MJ):
    os.makedirs(os.path.dirname(MJ), exist_ok=True)
    run(["curl", "-fsSL", "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js", "-o", MJ])
    print("vendored MathJax:", os.path.getsize(MJ) // 1024, "KB")

# ---------------------------------------------------------------- 1. figures
if os.path.isdir(FIGDIR):
    shutil.rmtree(FIGDIR)
os.makedirs(FIGDIR, exist_ok=True)
srcfig = os.path.join(PAPER, "figures")
for f in sorted(os.listdir(srcfig)):
    if f.endswith(".pdf"):
        run(["pdftocairo", "-svg", os.path.join(srcfig, f), os.path.join(FIGDIR, f[:-4] + ".svg")])
    elif f.endswith(".svg"):
        shutil.copy2(os.path.join(srcfig, f), os.path.join(FIGDIR, f))
print("figures -> svg:", len([f for f in os.listdir(FIGDIR) if f.endswith('.svg')]))

# ---------------------------------------------------------------- 2. numbers (.aux)
if os.path.isdir(AUXDIR):
    shutil.rmtree(AUXDIR)
os.makedirs(AUXDIR, exist_ok=True)
run(["tectonic", "-X", "compile", os.path.join(PAPER, TEX), "--outdir", AUXDIR,
     "--keep-intermediates", "--synctex=0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
aux = ""
for fn in os.listdir(AUXDIR):
    if fn.endswith(".aux"):
        aux += open(os.path.join(AUXDIR, fn)).read()
NUM = {}
for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', aux):
    lab, num = m.group(1), m.group(2)
    if lab not in NUM and re.match(r'^[0-9]', num):
        NUM[lab] = num
print("labels numbered:", len(NUM))

# ---------------------------------------------------------------- 3. pandoc
if os.path.isdir(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP, exist_ok=True)
for f in [TEX] + INPUTS:
    src = os.path.join(PAPER, f)
    if os.path.exists(src):
        t = open(src).read()
        t = t.replace(r"\begin{table*}", r"\begin{table}").replace(r"\end{table*}", r"\end{table}")
        open(os.path.join(TMP, f), "w").write(t)
run(["pandoc", TEX, "--from=latex", "--to=html5", "--mathjax", "--standalone",
     "--template=" + os.path.join(HERE, "template.html"),
     "--toc", "--toc-depth=3", "--number-sections", "--section-divs", "--shift-heading-level-by=1",
     "--citeproc", "--bibliography=" + os.path.join(PAPER, "refs.bib"),
     "-o", OUT], cwd=TMP)
h = open(OUT).read()

# ---------------------------------------------------------------- 4. post-process
# protect the TOC nav from cross-ref rewriting
mnav = re.search(r'<nav id="TOC".*?</nav>', h, re.S)
nav = mnav.group(0) if mnav else ""
if nav:
    h = h.replace(nav, "@@TOC@@")

WORD = {'tab': ('Table', 'Tables'), 'fig': ('Figure', 'Figures')}

def join_links(items):
    if len(items) == 1: return items[0]
    if len(items) == 2: return items[0] + ' and ' + items[1]
    return ', '.join(items[:-1]) + ' and ' + items[-1]

# 4b. single-label refs (no comma in href): rewrite the visible text using NUM
def single_repl(m):
    lab = m.group('lab'); pref = lab.split(':', 1)[0]; n = NUM.get(lab)
    if n is None:
        return m.group(0)
    if pref == 'tab': txt = f'Table&nbsp;{n}'
    elif pref == 'fig': txt = f'Figure&nbsp;{n}'
    else:  # section: keep a single § (source often already prints one via \S)
        pre = h[max(0, m.start() - 3):m.start()]
        txt = n if '§' in pre else '§' + n
    return f'<a class="xref" href="#{lab}">{txt}</a>'
h = re.sub(r'<a\b[^>]*?href="#(?P<lab>(?:tab|fig|sec):[^",]+)"[^>]*?>(?:\[[^\]]*\]|[^<]*)</a>',
           single_repl, h, flags=re.S)

# 4a. multi-label refs: one <a> whose href is "#a,b,c" -> a real list of links
def multi_repl(m):
    labels = [x.strip() for x in m.group('ref').split(',')]
    parts = []
    for pref, grp in groupby(labels, key=lambda l: l.split(':', 1)[0]):
        g = list(grp)
        links = [f'<a class="xref" href="#{l}">{("§"+NUM.get(l,"?")) if pref=="sec" else NUM.get(l,"?")}</a>' for l in g]
        if pref in WORD:
            parts.append((WORD[pref][0] if len(g) == 1 else WORD[pref][1]) + '&nbsp;' + join_links(links))
        else:
            parts.append(join_links(links))
    return ' '.join(parts)
h = re.sub(r'<a\b[^>]*?href="#(?P<ref>(?:tab|fig|sec):[^"]*,[^"]*)"[^>]*?>\[[^\]]*\]</a>',
           multi_repl, h, flags=re.S)

# 4c. tables: <div id="tab:X"><table>..<caption>C</caption>..</table></div>
#     -> caption line above a horizontal-scroll wrapper, numbered "Table N."
def table_repl(m):
    lab = m.group('lab'); inner = m.group('inner'); n = NUM.get(lab, '?')
    capm = re.search(r'<caption>(.*?)</caption>', inner, re.S)
    cap = capm.group(1).strip() if capm else ''
    tbl = re.sub(r'<caption>.*?</caption>', '', inner, flags=re.S)
    capdiv = f'<div class="tcap"><span class="tnum">Table&nbsp;{n}.</span> {cap}</div>'
    return f'<div id="{lab}" class="tblock">{capdiv}<div class="tscroll">{tbl}</div></div>'
h = re.sub(r'<div id="(?P<lab>tab:[^"]+)">\s*(?P<inner><table.*?</table>)\s*</div>',
           table_repl, h, flags=re.S)

# 4e. figures: number the caption "Figure N."
def fig_repl(m):
    lab = m.group('lab'); n = NUM.get(lab, '?')
    return m.group(0).replace('<figcaption>', f'<figcaption><span class="tnum">Figure&nbsp;{n}.</span> ', 1)
h = re.sub(r'<figure id="(?P<lab>fig:[^"]+)".*?</figure>', fig_repl, h, flags=re.S)

# Pandoc drops the body of inline TikZ figures.  Keep TikZ authoritative for PDF,
# but inject the checked-in accessible SVG equivalent into the HTML figure.
def design_repl(m):
    figure = m.group(0)
    if '<img ' not in figure:
        figure = figure.replace('>',
            '><img src="figures/study_design.svg" alt="Five-stage clean-rerun study design" loading="lazy"/>',
            1)
    return figure
h = re.sub(r'<figure id="fig:design".*?</figure>', design_repl, h, flags=re.S)

# 4f. figure images -> SVG, and pandoc's <embed> (used for PDFs) -> responsive <img>
h = re.sub(r'(figures/[A-Za-z0-9_]+)\.pdf', r'\1.svg', h)
def embed_repl(m):
    src = m.group('src')
    if src.endswith('specialization_plane.svg'):
        alt = ('Scatter plot of twenty seed-level represented-source and transfer AP changes; '
               'fourteen points are in the specialization quadrant and six in uniform gain.')
    else:
        alt = 'Paper figure'
    return f'<img src="{src}" alt="{alt}" loading="lazy"/>'

h = re.sub(r'<embed\s+src="(?P<src>figures/[^"]+)"\s*/?>', embed_repl, h)

# Pandoc can preserve a LaTeX equation environment *inside* its own \[...\]
# display wrapper.  MathJax rejects that nested display structure.  Keep the
# equation label as an HTML anchor and pass only the equation body to MathJax.
def equation_repl(m):
    label = m.group('label')
    body = m.group('body').strip()
    return f'<span id="{label}" class="math display">\\[{body}\\]</span>'

h = re.sub(
    r'<span\s+class="math display">\\\[\s*\\begin\{equation\*?\}\s*'
    r'\\label\{(?P<label>[^}]+)\}\s*(?P<body>.*?)\s*'
    r'\\end\{equation\*?\}\s*\\\]</span>',
    equation_repl,
    h,
    flags=re.S,
)

nested_equations = len(re.findall(
    r'<span[^>]*class="math display"[^>]*>\\\[\s*\\begin\{equation', h
))
if nested_equations:
    raise RuntimeError(f"MathJax-incompatible nested equation wrappers: {nested_equations}")
# Other math stays as TeX in \(...\) / \[...\] for offline MathJax-SVG.

# restore TOC
if nav:
    h = h.replace("@@TOC@@", nav)

open(OUT, "w").write(h)
print("wrote", OUT, f"({len(h)//1024} KB)")
print("tables:", h.count('class="tblock"'), " figures:", h.count('class="tnum">Figure'),
      " xrefs:", h.count('class="xref"'), " leftover brackets:",
      len(re.findall(r'\[(?:tab|fig|sec):[a-z0-9,:_-]+\]', h)),
      " nested equations:", nested_equations)
