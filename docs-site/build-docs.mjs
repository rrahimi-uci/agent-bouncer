/**
 * CALIBER documentation site generator.
 *
 * Reads the architecture series of markdown files under `docs/` and renders each
 * into a polished, self-navigating HTML page under `docs-site/`, reusing the
 * shared design system (`docs.css` + `docs.js`). It also emits `docs-nav.js`,
 * the single source of truth for the sidebar navigation shared by every page
 * (including the hand-authored landing `index.html`).
 *
 * The renderer is intentionally dependency-free so this can run inside the
 * `prebuild` hook in any context (including Docker stages with no node_modules).
 * It supports exactly the markdown constructs the docs use: ATX headings, GFM
 * tables, fenced code (with `mermaid` diagrams rendered client-side and
 * `diagram-svg` assets inlined at build time), ordered and unordered lists with
 * nesting + wrapped continuations, blockquotes (which may themselves contain
 * tables), and inline code / bold / italic / links.
 *
 * Cross-references between docs (`../11-test-sets/architecture.md`) are rewritten
 * to the generated page; links into source files are downgraded to inline code
 * so the docs never carry a broken hyperlink.
 *
 * Usage:  node docs-site/build-docs.mjs
 */

import { readFileSync, writeFileSync, readdirSync, existsSync, rmSync, renameSync } from "node:fs";
import { dirname, relative, resolve, posix } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = resolve(here, "../docs"); // agent-bouncer/docs
const OUT_DIR = here; // agent-bouncer/docs-site
const BRAND_SHORT = "Agent Bouncer";
const BRAND_FULL = "Agent Bouncer — a tiny, fast SLM safety guardrail for LLMs & agents";
const DOCS_HOME_LABEL = `${BRAND_FULL} docs home`;

function writeTextAtomic(dest, contents) {
  const tmp = `${dest}.tmp-${process.pid}-${Date.now()}`;
  writeFileSync(tmp, contents, "utf8");
  try {
    renameSync(tmp, dest);
  } catch (err) {
    try {
      rmSync(tmp, { force: true });
    } catch {}
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Module manifest — ordering, grouping, output filenames, and nav labels.
// This is the single source of truth the generator and the sidebar share.
// ---------------------------------------------------------------------------

const GROUPS = [
  { id: "learn", title: "Learn" },
  { id: "studio", title: "Studio & workflow" },
  { id: "benchmarks", title: "Benchmarks & results" },
  { id: "architecture", title: "Architecture" },
  { id: "strategy", title: "Roadmap" },
];

const MODULES = [
  { md: "slm-architectures.md", out: "m-slm-architectures.html", group: "learn", label: "SLM architectures", blurb: "Encoder vs decoder guardrails, the modern decoder block (RMSNorm · RoPE · GQA · SwiGLU), and a deep dive into each base model — DistilBERT, ModernBERT, Qwen3-0.6B/1.7B, DeepSeek-R1-Distill, SmolLM2." },
  { md: "fine-tuning.md", out: "m-fine-tuning.html", group: "learn", label: "Fine-tuning techniques", blurb: "SFT · LoRA · GRPO (RLVR) · DPO explained with diagrams — what each technique optimizes, when to reach for it, and which technique applies to which base model." },
  { md: "taxonomy.md", out: "m-taxonomy.html", group: "learn", label: "Safety taxonomy", blurb: "The hazard taxonomy the guards label against — the safe/unsafe decision, hazard categories, and how benchmark labels map onto them (positive class = unsafe)." },
  { md: "workflow.md", out: "m-workflow.html", group: "studio", label: "Guided workflow", blurb: "The end-to-end loop in the Benchmark Studio: explore benchmarks → build a leakage-free train/test set → pick model × technique → train → test → save → evaluate & compare on the leaderboard." },
  { md: "datasets.md", out: "m-datasets.html", group: "studio", label: "Datasets", blurb: "How training sets are composed from benchmark sources, the balancing strategies, and the disjoint train/test split with an enforced no-leakage guarantee." },
  { md: "benchmarks.md", out: "m-benchmarks.html", group: "benchmarks", label: "Benchmark suite", blurb: "The seven-benchmark standard suite across guardrail, red-teaming, and over-refusal axes; every guard scored through one harness; over-blocking (FPR@benign) as the headline usability metric." },
  { md: "ensembles.md", out: "m-ensembles.html", group: "benchmarks", label: "Ensembles", blurb: "Combining guards into one — union / intersection / majority / mean / weighted strategies — evaluated offline from dumped per-sample predictions, and the interactive ensemble builder in the Studio." },
  { md: "architecture.md", out: "m-architecture.html", group: "architecture", label: "System architecture", blurb: "How Agent Bouncer fits together: the guard interface, the FastAPI Benchmark Studio, the evaluation harness, training + versioning, experiment tracking, and the model store." },
  { md: "roadmap.md", out: "m-roadmap.html", group: "strategy", label: "Roadmap", blurb: "Where Agent Bouncer is headed — planned guards, benchmarks, training techniques, and Studio features." },
];

// Fast lookup: normalized "<dir>/<file>.md" (relative to docs/) -> output html.
const mdToHtml = new Map(MODULES.map((m) => [m.md, m.out]));

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function leadingSpaces(line) {
  const m = line.match(/^(\s*)/);
  return m ? m[1].replace(/\t/g, "    ").length : 0;
}

function resolveDocAsset(ref, fromMdRel) {
  const assetPath = resolve(DOCS_DIR, posix.dirname(fromMdRel), ref);
  const rel = relative(DOCS_DIR, assetPath);
  if (!rel || rel === "." || rel.startsWith("..")) {
    throw new Error(`[build-docs] asset ${ref} in ${fromMdRel} resolves outside docs/`);
  }
  if (!existsSync(assetPath)) {
    throw new Error(`[build-docs] missing asset ${ref} referenced from ${fromMdRel}`);
  }
  return assetPath;
}

function renderSvgDiagramAsset(ref, fromMdRel) {
  const assetRef = ref.trim();
  if (!assetRef) {
    throw new Error(`[build-docs] empty diagram-svg fence in ${fromMdRel}`);
  }
  const assetPath = resolveDocAsset(assetRef, fromMdRel);
  const raw = readFileSync(assetPath, "utf8").replace(/^\uFEFF/, "");
  const match = raw.match(/<svg\b[\s\S]*<\/svg>/i);
  if (!match) {
    throw new Error(`[build-docs] diagram-svg asset ${assetRef} in ${fromMdRel} does not contain a root <svg>`);
  }
  const svg = match[0].trim();
  if (/<script\b/i.test(svg) || /\son[a-z]+\s*=/i.test(svg)) {
    throw new Error(`[build-docs] diagram-svg asset ${assetRef} in ${fromMdRel} must be a static SVG`);
  }
  return `<figure class="diagram diagram-svg" data-diagram-src="${escapeAttr(assetRef)}">${svg}</figure>`;
}

// ---------------------------------------------------------------------------
// Link resolution — cross-doc refs become page links; source files become code.
// ---------------------------------------------------------------------------

function classifyLink(href, fromMdRel) {
  const [path, ...hashParts] = href.split("#");
  const hash = hashParts.length ? "#" + hashParts.join("#") : "";

  if (/^https?:\/\//.test(href) || href.startsWith("mailto:")) {
    return { kind: "external", href };
  }
  if (path === "" && hash) {
    return { kind: "anchor", href: hash };
  }
  if (/\.md$/.test(path)) {
    // Resolve relative to the source file's directory, then to docs/ root.
    const fromDir = posix.dirname(fromMdRel);
    const rel = posix.normalize(posix.join(fromDir, path));
    const out = mdToHtml.get(rel);
    if (out) return { kind: "page", href: out + hash };
    return { kind: "code" }; // unknown doc — render as code, never a broken link
  }
  // Source files and any other repo-relative path: show as code, not a link.
  return { kind: "code" };
}

// ---------------------------------------------------------------------------
// Inline rendering (code spans, escaping, links, bold, italic)
// ---------------------------------------------------------------------------

function renderInline(text, fromMdRel) {
  const codeSpans = [];
  // 1. Pull out inline code first so nothing inside it is reinterpreted. The
  //    @@C<n>@@ sentinel survives HTML escaping and the emphasis passes and
  //    adds no whitespace around code that abuts punctuation, e.g. `server.py`.
  let work = text.replace(/`([^`]+)`/g, (_, code) => {
    const i = codeSpans.length;
    codeSpans.push(`<code>${escapeHtml(code)}</code>`);
    return `@@C${i}@@`;
  });

  // 2. Escape the rest of the HTML-significant characters.
  work = escapeHtml(work);

  // 3. Bold then italic (asterisk form only — underscores appear in identifiers).
  work = work.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  work = work.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

  // 4. Links: [label](href). The label is already escaped + emphasized; a link
  //    into a source file is downgraded to its (often code-formatted) label so
  //    the docs never carry a broken hyperlink.
  work = work.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, rawHref) => {
    const info = classifyLink(rawHref.trim(), fromMdRel);
    if (info.kind === "code") return label;
    if (info.kind === "external") {
      return `<a href="${escapeAttr(info.href)}" target="_blank" rel="noopener">${label}</a>`;
    }
    return `<a href="${escapeAttr(info.href)}">${label}</a>`;
  });

  // 5. Restore code spans.
  work = work.replace(/@@C(\d+)@@/g, (_, n) => codeSpans[Number(n)]);
  return work;
}

// The diagram color key. Emitted by an empty ```legend``` fence. Dot classes
// (.legend-dot.user/.ui/.ctrl/.store/.ext/.async) are styled in docs.css and
// mirror the semantic node classDefs docs.js injects into flowcharts.
const LEGEND_HTML = `<div class="diagram-legend" role="note" aria-label="Diagram color key">
  <span class="legend-item"><span class="legend-dot user"></span>Actor</span>
  <span class="legend-item"><span class="legend-dot ui"></span>UI / SPA</span>
  <span class="legend-item"><span class="legend-dot ctrl"></span>Control plane</span>
  <span class="legend-item"><span class="legend-dot store"></span>Storage</span>
  <span class="legend-item"><span class="legend-dot ext"></span>External</span>
  <span class="legend-item"><span class="legend-dot async"></span>Async worker</span>
</div>`;

// ---------------------------------------------------------------------------
// Block rendering
// ---------------------------------------------------------------------------

function splitTableRow(row) {
  const PIPE = "\u0001";
  let s = row.trim().replace(/\\\|/g, PIPE); // protect escaped pipes
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.split(PIPE).join("|").trim());
}

function isTableSeparator(line) {
  return (
    line.includes("-") &&
    /^\s*\|?[\s:|-]+\|?\s*$/.test(line) &&
    line.includes("|")
  );
}

function renderTable(headerCells, alignRow, rows, fromMdRel) {
  const aligns = alignRow.map((c) => {
    const left = c.startsWith(":");
    const right = c.endsWith(":");
    if (left && right) return "center";
    if (right) return "right";
    if (left) return "left";
    return "";
  });
  const th = headerCells
    .map((c, idx) => {
      const a = aligns[idx] ? ` style="text-align:${aligns[idx]}"` : "";
      return `<th${a}>${renderInline(c, fromMdRel)}</th>`;
    })
    .join("");
  const body = rows
    .map((cells) => {
      const tds = cells
        .map((c, idx) => {
          const a = aligns[idx] ? ` style="text-align:${aligns[idx]}"` : "";
          return `<td${a}>${renderInline(c, fromMdRel)}</td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}

// Parse a list (ordered or unordered) starting at `start`. Handles wrapped
// continuation lines and nested sublists via indentation. Returns rendered HTML
// and the index of the first line after the list.
function parseList(lines, start, fromMdRel) {
  const firstIndent = leadingSpaces(lines[start]);
  const ordered = /^\s*\d+\.\s/.test(lines[start]);
  const tag = ordered ? "ol" : "ul";
  const items = [];
  let i = start;

  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*$/.test(line)) {
      const next = lines[i + 1];
      if (next && !/^\s*$/.test(next) && leadingSpaces(next) > firstIndent) {
        i++; // blank line inside an item's nested block
        continue;
      }
      break;
    }
    const indent = leadingSpaces(line);
    const m = line.match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
    if (!m || indent !== firstIndent) break;

    let content = m[3];
    i++;
    const childLines = [];
    while (i < lines.length) {
      const l = lines[i];
      if (/^\s*$/.test(l)) {
        const n = lines[i + 1];
        if (n && !/^\s*$/.test(n) && leadingSpaces(n) > firstIndent) {
          childLines.push("");
          i++;
          continue;
        }
        break;
      }
      if (leadingSpaces(l) <= firstIndent) break;
      childLines.push(l.slice(firstIndent));
      i++;
    }
    items.push({ content, childLines });
  }

  const lis = items
    .map((it) => {
      // Leading non-marker child lines are wrapped continuations of the item.
      let k = 0;
      const cont = [];
      while (
        k < it.childLines.length &&
        it.childLines[k] !== "" &&
        !/^\s*([-*+]|\d+\.)\s+/.test(it.childLines[k])
      ) {
        cont.push(it.childLines[k].trim());
        k++;
      }
      const merged = (it.content + " " + cont.join(" ")).trim();
      let inner = renderInline(merged, fromMdRel);
      const rest = it.childLines.slice(k);
      if (rest.some((l) => l.trim() !== "")) {
        inner += renderBlocks(rest.join("\n"), fromMdRel);
      }
      return `<li>${inner}</li>`;
    })
    .join("");

  return { html: `<${tag}>${lis}</${tag}>`, next: i };
}

function renderBlocks(md, fromMdRel) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (/^\s*$/.test(line)) {
      i++;
      continue;
    }

    // Fenced code block
    const fence = line.match(/^\s*```\s*([\w-]*)\s*$/);
    if (fence) {
      const lang = fence[1].trim().toLowerCase();
      i++;
      const buf = [];
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
        buf.push(lines[i]);
        i++;
      }
      i++; // closing fence
      const code = buf.join("\n");
      if (lang === "mermaid") {
        // Authors sometimes write a literal `\n` for a node-label line break, but
        // Mermaid expects `<br/>` (a bare `\n` collapses the label onto one line).
        // Normalize before escaping; docs.js restores the escaped tag via
        // textContent at render time so Mermaid receives a real `<br/>`.
        const diagram = code.replace(/ *\\n */g, "<br/>");
        out.push(`<div class="diagram"><pre class="mermaid">${escapeHtml(diagram)}</pre></div>`);
      } else if (lang === "diagram-svg") {
        // Presentation-grade diagrams live as checked-in SVG assets near the
        // doc that uses them. We inline the exported SVG so theme variables from
        // docs.css can restyle it live without adding a new asset pipeline.
        out.push(renderSvgDiagramAsset(code, fromMdRel));
      } else if (lang === "legend") {
        // An empty ```legend``` fence renders the shared diagram color key. The
        // dot colors are defined (theme-aware) in docs.css and mirror the
        // semantic classDefs docs.js injects into flowcharts.
        out.push(LEGEND_HTML);
      } else {
        out.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
      }
      continue;
    }

    // Heading
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const raw = h[2].trim().replace(/\s+#+\s*$/, "");
      const id = slugify(raw);
      out.push(
        `<h${level} id="${id}">${renderInline(raw, fromMdRel)}` +
          `<a class="anchor" href="#${id}" aria-hidden="true" tabindex="-1">#</a></h${level}>`
      );
      i++;
      continue;
    }

    // Blockquote (may contain block content, e.g. a table)
    if (/^\s*>/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      out.push(`<blockquote class="callout">${renderBlocks(buf.join("\n"), fromMdRel)}</blockquote>`);
      continue;
    }

    // GFM table
    if (
      line.includes("|") &&
      i + 1 < lines.length &&
      isTableSeparator(lines[i + 1])
    ) {
      const headerCells = splitTableRow(line);
      const alignRow = splitTableRow(lines[i + 1]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "") {
        rows.push(splitTableRow(lines[i]));
        i++;
      }
      out.push(renderTable(headerCells, alignRow, rows, fromMdRel));
      continue;
    }

    // List
    if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
      const { html, next } = parseList(lines, i, fromMdRel);
      out.push(html);
      i = next;
      continue;
    }

    // Paragraph — gather consecutive plain lines.
    const buf = [line];
    i++;
    while (
      i < lines.length &&
      !/^\s*$/.test(lines[i]) &&
      !/^(#{1,6})\s/.test(lines[i]) &&
      !/^\s*```/.test(lines[i]) &&
      !/^\s*>/.test(lines[i]) &&
      !/^\s*([-*+]|\d+\.)\s+/.test(lines[i]) &&
      !(lines[i].includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1]))
    ) {
      buf.push(lines[i]);
      i++;
    }
    out.push(`<p>${renderInline(buf.join(" "), fromMdRel)}</p>`);
  }

  return out.join("\n");
}

// ---------------------------------------------------------------------------
// Reference tier — split each page into an Overview tier (everything above the
// `## Reference` heading) and a banded, progressively-disclosed Reference tier.
// This runs on the already-rendered body HTML so the core markdown parser stays
// untouched. If a doc has no `## Reference` marker, the body is returned as-is.
// ---------------------------------------------------------------------------

function applyReferenceTier(html) {
  const marker = html.search(/<h2 id="reference">/);
  if (marker === -1) return html;

  const before = html.slice(0, marker);
  const rest = html.slice(marker);

  // The marker heading itself becomes a banded tier header; preserve its id and
  // visible text so the TOC entry and #reference anchor keep working.
  const head = rest.match(/^<h2 id="reference">([\s\S]*?)(<a class="anchor"[\s\S]*?)?<\/h2>/);
  const title = head ? head[1].trim() : "Reference";
  const afterHeadIdx = rest.indexOf("</h2>") + "</h2>".length;
  const afterRef = rest.slice(afterHeadIdx);

  const banner =
    `<div class="ref-tier-header">` +
    `<span class="ref-tier-eyebrow">Deep reference · data models, APIs &amp; lifecycle</span>` +
    `<h2 id="reference" class="ref-tier-title">${title}` +
    `<a class="anchor" href="#reference" aria-hidden="true" tabindex="-1">#</a></h2>` +
    `</div>`;

  // Wrap each `##` section below the marker in a default-open <details> so the
  // deep tier is skimmable but collapsible. Content before the first such
  // heading (a lead paragraph under `## Reference`) renders outside the panels.
  let wrapped = "";
  for (const part of afterRef.split(/(?=<h2 )/)) {
    if (!part.trim()) continue;
    if (part.startsWith("<h2 ")) {
      const e = part.indexOf("</h2>") + "</h2>".length;
      const summary = part.slice(0, e);
      const body = part.slice(e).trim();
      wrapped +=
        `<details class="ref-section" open>` +
        `<summary class="ref-section-summary">${summary}</summary>` +
        `<div class="ref-section-body">${body}</div></details>`;
    } else {
      wrapped += `<div class="ref-tier-lead">${part}</div>`;
    }
  }

  return `${before}<section class="ref-tier">${banner}${wrapped}</section>`;
}

// ---------------------------------------------------------------------------
// Page assembly
// ---------------------------------------------------------------------------

const THEME_BOOT = `<script>
    (function () {
      try {
        var t = localStorage.getItem("agent-bouncer-docs-theme");
        if (t !== "light" && t !== "dark") {
          t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        }
        document.documentElement.dataset.theme = t;
      } catch (e) {
        document.documentElement.dataset.theme = "light";
      }
    })();
  </script>`;

function topbar() {
  return `<header class="topbar">
    <div class="topbar-row">
      <a class="topbar-brand" href="index.html" aria-label="${escapeAttr(DOCS_HOME_LABEL)}" title="${escapeAttr(DOCS_HOME_LABEL)}" style="text-decoration:none;color:inherit">
        <img src="agent-bouncer.png" alt="${escapeAttr(BRAND_SHORT)} logo">
        <strong>${escapeHtml(BRAND_SHORT)}</strong>
      </a>
      <nav class="docs-section-tabs" id="docsSectionTabs" aria-label="Documentation sections"></nav>
      <div class="topbar-actions">
        <button type="button" class="topbar-search-trigger" id="topbarSearch" aria-label="Search docs" title="Search docs">
          <svg class="topbar-search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
          <span class="topbar-search-label">Search docs</span>
          <span class="keycap" data-doc-search-key>⌘K</span>
        </button>
        <a class="topbar-cta" href="index.html#quickstart">Quickstart</a>
        <button type="button" class="theme-toggle" id="themeToggle" aria-label="Toggle dark mode" title="Toggle light / dark theme">
          <svg class="icon-moon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
          <svg class="icon-sun" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" /><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" /></svg>
        </button>
      </div>
    </div>
  </header>`;
}

function pageHtml({ title, groupTitle, label, bodyHtml }) {
  return `<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)} | ${escapeHtml(BRAND_SHORT)}</title>
  <meta name="description" content="${escapeAttr(`${title} — part of the ${BRAND_SHORT} documentation.`)}">
  <link rel="icon" type="image/png" href="agent-bouncer.png">
  ${THEME_BOOT}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="docs.css">
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
</head>

<body data-doc-page="module">
  <a class="skip-link" href="#docsContent">Skip to content</a>
  ${topbar()}

  <button type="button" class="menu-toggle" id="menuToggle" aria-label="Toggle menu" aria-controls="docsSidebar" aria-expanded="false">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  </button>

  <div class="layout">
    <aside class="sidebar" id="docsSidebar">
      <a class="sidebar-brand" href="index.html" aria-label="${escapeAttr(DOCS_HOME_LABEL)}" title="${escapeAttr(DOCS_HOME_LABEL)}" style="text-decoration:none;color:inherit">
        <img src="agent-bouncer.png" alt="${escapeAttr(BRAND_SHORT)} logo">
        <div class="sidebar-brand-title">${escapeHtml(BRAND_SHORT)}</div>
      </a>
      <div class="sidebar-search">
        <input id="nav-filter" type="search" placeholder="Search docs and sections..." aria-label="Search docs and sections">
      </div>
      <nav id="docs-nav" aria-label="Documentation"></nav>
    </aside>

    <main class="main" id="docsContent">
      <article class="content doc-body">
        <nav class="doc-breadcrumb" aria-label="Breadcrumb">
          <span>${escapeHtml(groupTitle)}</span>
          <span aria-hidden="true">/</span>
          <span class="current">${escapeHtml(label)}</span>
        </nav>
        ${bodyHtml}
        <footer>
          <p>
            ${escapeHtml(BRAND_FULL)} —
            this page is generated from the architecture series in <code>docs/</code>.
          </p>
        </footer>
      </article>
    </main>

    <aside class="toc" aria-label="On this page">
      <div class="toc-card">
        <div class="toc-title">On this page</div>
        <nav class="toc-links" id="page-toc"></nav>
      </div>
    </aside>
  </div>

  <script defer src="docs-nav.js"></script>
  <script defer src="docs.js"></script>
</body>

</html>
`;
}

// Render one markdown module to a full HTML page.
function renderModule(mod) {
  const srcPath = resolve(DOCS_DIR, mod.md);
  const raw = readFileSync(srcPath, "utf8");
  const lines = raw.replace(/\r\n/g, "\n").split("\n");

  // Pull out the first H1 as the page title; render the rest as the body.
  let title = mod.label;
  let startIdx = 0;
  for (let j = 0; j < lines.length; j++) {
    if (/^\s*$/.test(lines[j])) continue;
    const h1 = lines[j].match(/^#\s+(.*)$/);
    if (h1) {
      title = h1[1].trim();
      startIdx = j + 1;
    }
    break;
  }
  const bodyMd = lines.slice(startIdx).join("\n");
  const group = GROUPS.find((g) => g.id === mod.group);
  const groupTitle = group ? group.title : "Architecture";
  const blurb = typeof mod.blurb === "string" ? mod.blurb.trim() : "";
  const eyebrow = groupTitle;

  const header = `<header class="doc-header">
          <div class="doc-eyebrow">${escapeHtml(eyebrow)}</div>
          <h1 id="top">${escapeHtml(title)}</h1>
          ${blurb ? `<p class="doc-summary">${escapeHtml(blurb)}</p>` : ""}
          <div class="doc-actions">
            <button type="button" class="doc-copy-button" data-copy-page data-copy-default="Copy page" data-copy-success="Copied" data-copy-failure="Copy failed">
              Copy page
            </button>
          </div>
        </header>`;

  const bodyHtml = header + "\n" + applyReferenceTier(renderBlocks(bodyMd, mod.md));
  return pageHtml({ title, groupTitle, label: mod.label, bodyHtml });
}

// Emit docs-nav.js — the shared sidebar definition consumed by docs.js. Only the
// modules whose source actually exists are listed, so a deleted .md drops out of
// the nav automatically rather than leaving a dangling link.
function buildNavData(present) {
  const sections = [
    { section: "Documentation", links: [{ href: "index.html", label: "Overview" }] },
  ];
  for (const g of GROUPS) {
    const links = present
      .filter((m) => m.group === g.id)
      .map((m) => ({ href: m.out, label: m.label }));
    if (links.length) sections.push({ section: g.title, links });
  }
  return sections;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

function main() {
  if (!existsSync(DOCS_DIR)) {
    console.log(`[build-docs] ${DOCS_DIR} not found — nothing to generate.`);
    return;
  }

  const present = [];
  for (const mod of MODULES) {
    const src = resolve(DOCS_DIR, mod.md);
    if (!existsSync(src)) {
      console.warn(`[build-docs] missing source ${mod.md} — skipping ${mod.out}`);
      const staleOut = resolve(OUT_DIR, mod.out);
      if (existsSync(staleOut)) {
        rmSync(staleOut);
        console.warn(`[build-docs] removed stale page ${mod.out}`);
      }
      continue;
    }
    const html = renderModule(mod);
    if (!html.trim() || !html.includes("<!DOCTYPE html>")) {
      throw new Error(`[build-docs] refusing to write invalid output for ${mod.out}`);
    }
    writeTextAtomic(resolve(OUT_DIR, mod.out), html);
    // Emit the raw Markdown alongside each page (e.g. m-01-platform.md) so
    // llms.txt can point at a clean, agent-consumable source — the doc is
    // literally about agentic workflows, so this matters.
    writeTextAtomic(resolve(OUT_DIR, mod.out.replace(/\.html$/, ".md")), readFileSync(resolve(DOCS_DIR, mod.md), "utf8"));
    present.push(mod);
  }

  const nav = buildNavData(present);
  const navJs =
    "/* Generated by build-docs.mjs — do not edit by hand. */\n" +
    "window.DOCS_NAV = " +
    JSON.stringify(nav, null, 2) +
    ";\n";
  writeTextAtomic(resolve(OUT_DIR, "docs-nav.js"), navJs);

  // llms.txt — a machine index of the documentation (https://llmstxt.org/). Links
  // point at the raw .md siblings (clean source, no chrome) so an agent can read
  // the docs programmatically. Grouped to mirror the sidebar.
  const llmsLines = [
    "# Agent Bouncer",
    "",
    "> A tiny, fast small-language-model (SLM) safety guardrail for LLMs and AI agents — screens prompts, tool calls, and outputs for prompt injection, jailbreaks, and unsafe content, trained with fine-tuning + RL and benchmarked against GPT-4o-mini, GPT-5.2, and OpenAI Moderation through one honest harness.",
    "",
    "The pages below are the documentation, one per topic. Each link is the raw Markdown source, built for programmatic access.",
    "",
  ];
  for (const g of GROUPS) {
    const mods = present.filter((m) => m.group === g.id);
    if (!mods.length) continue;
    llmsLines.push(`## ${g.title}`, "");
    for (const m of mods) {
      const mdName = m.out.replace(/\.html$/, ".md");
      const blurb = typeof m.blurb === "string" ? m.blurb.trim().replace(/\s+/g, " ") : "";
      llmsLines.push(`- [${m.label}](${mdName})${blurb ? ": " + blurb : ""}`);
    }
    llmsLines.push("");
  }
  writeTextAtomic(resolve(OUT_DIR, "llms.txt"), llmsLines.join("\n"));

  console.log(`[build-docs] generated ${present.length} module pages + ${present.length} .md + docs-nav.js + llms.txt`);
}

main();

export { renderBlocks, renderInline, MODULES };
