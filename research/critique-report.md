# Technical Critique of `research/`

Scope: static review of the committed `research/` bundle, with emphasis on LLM guardrail methodology, operating-point fairness, benchmark validity, and reproducibility. I did not re-run full training or paid API evaluations; findings are grounded in the shipped docs, scripts, and committed output artifacts.

## Findings

1. **Critical — the advertised “primary artifact” notebook is not the paper artifact, and it still carries the old GPT failure policy.**  
   Evidence: `research/README.md:21-32` calls the notebook the “primary artifact” and says it is fully self-contained for reproducing the paper. `research/notebooks/README.md:5-20` instead describes a mini-judge parity notebook built from `docs/smollm3-guard-plan.md`, not the paper’s open-guard / AUPRC / matched-FPR study. In code, `research/scripts/build_smollm3_notebook.py:542-570` evaluates `gpt-4o-mini` and `gpt-5-mini` under a non-inferiority framing, while `research/scripts/build_smollm3_notebook.py:524-539` still does `preds.append(1)` on OpenAI exceptions (“fail closed”).  
   Why this matters: a reader who follows the top-level reproduction path will not reproduce the paper’s central claims about Llama-Guard, ShieldGemma, matched-FPR, or threshold-free AUPRC. Worse, the shipped notebook generator still contains the very GPT error-handling bug the paper says was fixed.

2. **High — the claimed GPT baseline bug fix is not actually re-grounded in the corrected in-house evaluation path.**  
   Evidence: `research/scripts/eval_large_guard.py:277-289` generates GPT predictions with fail-closed error handling and writes them into `preds_large.json`. `research/scripts/eval_corrected.py:1-7` says the corrected eval fixes the GPT bug, but `research/scripts/eval_corrected.py:108-109` reloads `preds_large.json` and `research/scripts/eval_corrected.py:280-283` reuses those cached GPT hard predictions instead of re-calling GPT with abstention-aware logic. The committed `research/notebooks/outputs/nb-smollm3-guard/summary_large.json:83-89` records only hard GPT predictions, and `research/notebooks/outputs/nb-smollm3-guard/summary_corrected.json:20-45` contains no corrected GPT artifact at all.  
   Why this matters: if any API failures occurred in the original `eval_large_guard.py` run, the “corrected” GPT comparison still inherits the bias. Even if no failures occurred, the repo does not provide machine-readable evidence that the fix was actually applied to the paper’s GPT tie result.

3. **High — the hardened mortgage benchmark is internally inconsistent across the committed dataset, the builder, the results doc, and the paper.**  
   Evidence: `research/docs/mortgage-benchmark-hard-results.md:3-17` describes a 318-row dataset with a family-safe dev/test split, wrapper variants, and Recall@FPR reporting. `research/scripts/build_hard_jsonl.mjs:17-64` likewise builds a split 318-row artifact from `hard_admitted.json`. But `research/scripts/eval_mortgage_hard.py:5-8` says the committed set has no train/dev split or protected-class tags and evaluates it via threshold-free AUPRC plus per-model Optimal-F1; the committed output `research/notebooks/outputs/nb-smollm3-guard/summary_mortgage_hard.json:2-4,36-100` confirms a different 334-row artifact. The paper follows the 334-row version in `research/paper/benchmark_chooses_the_winner.tex:574-603`.  
   Why this matters: the reproduction chain in `research/README.md:35-45` is false as written. A reviewer cannot tell which mortgage benchmark actually underlies the load-bearing claims, and the shipped builder does not regenerate the benchmark the paper is analyzing.

4. **High — the mortgage case-study conclusions abandon the paper’s own operating-point-fair standard, then make strong frontier and fairness claims anyway.**  
   Evidence: the paper’s core thesis is operating-point fairness (`research/paper/benchmark_chooses_the_winner.tex:31-34`, `80-85`). But the mortgage script explicitly switches to per-model Optimal-F1 / best-threshold evaluation for local models and native GPT for the frontier baseline (`research/scripts/eval_mortgage_hard.py:5-8`, `103-114`, `136-141`). The paper then argues that the frontier “holds FPR at 0.065 at the same recall” and that the winner flips benchmark-dependently (`research/paper/benchmark_chooses_the_winner.tex:598-603`).  
   Why this matters: those FPR and ranking claims are not apples-to-apples. Threshold-free AUPRC can support the narrower claim that the hardened set de-saturates the comparison, but the stronger “capability gap,” “frontier ceiling,” and benchmark-inversion conclusions are materially weaker when each model is read at a different operating point.

5. **Medium — provenance claims around the figures are contradictory, and the figure script does not actually regenerate from result JSON.**  
   Evidence: `research/README.md:53-54` and `research/paper/README.md:9-10,27-40` say the paper numbers trace to committed JSON and that figures regenerate from result JSON. But `research/scripts/README.md:23` says `make_figures.py` uses “numbers inline,” and `research/scripts/make_figures.py:23-81` hardcodes the paper values directly without loading any JSON.  
   Why this matters: figure regeneration is not provenance-preserving. The script can silently drift from the committed outputs or the TeX tables, which is exactly the opposite of what a measurement paper should optimize for.

6. **Medium — the bundle is not actually self-contained or immutable, despite repeatedly claiming that it is.**  
   Evidence: `research/README.md:3,21-22` says nothing depends on code or data outside `research/`, and `research/notebooks/README.md:8-18` says the only network requirement is a model download. In practice, `research/scripts/eval_corrected.py:29-70` reconstructs the in-house evaluation set from live Hugging Face datasets with no pinned revisions, and `research/scripts/verify_novel.py:18-31` reconstructs the novel benchmarks the same way.  
   Why this matters: reruns can drift or fail as upstream datasets change. That is especially problematic here because the paper’s credibility rests on exact row alignment, matched-n comparisons, and “same 2,018 / 2,020 rows” language.

7. **Medium — the mortgage section’s own supporting doc already says the benchmark is not ready for load-bearing paper claims, but the paper still uses it as a thesis-carrying result.**  
   Evidence: `research/docs/mortgage-benchmark-hard-results.md:29-33` says the set is small, model-family-circular, and still needs SME re-adjudication before “load-bearing paper claims.” The paper nonetheless elevates the mortgage case study into the abstract and contributions (`research/paper/benchmark_chooses_the_winner.tex:34`, `83`) and uses it to argue benchmark-dependent winner inversion (`research/paper/benchmark_chooses_the_winner.tex:598-603`).  
   Why this matters: this is not just a caveat; it changes how much argumentative weight the section can carry. In its current state the mortgage case study is better framed as exploratory evidence than as one of the paper’s three central results.

## Open Questions / Assumptions

- I assumed `preds_large.json` was generated by the committed `research/scripts/eval_large_guard.py`. If those predictions were later replaced by a manual abstention-aware rerun, the repo needs explicit lineage for that fact.
- I treated the committed `research/notebooks/data/benchmarks/full/guard_benchmark_hard.jsonl` and `summary_mortgage_hard.json` as the source of truth for the shipped bundle, since they are the artifacts a reviewer actually receives.
- I did not validate external citations or upstream benchmark licenses in detail; this critique is about methodological grounding and reproducibility of the shipped bundle.

## Recommended Corrections

1. Either regenerate the notebook to match the paper exactly, or stop calling it the paper’s “primary artifact.” Right now it is a different experiment.
2. Re-run the GPT baseline end-to-end with abstention logging, persist the corrected paired comparison in JSON, and stop relying on `preds_large.json` for the paper’s corrected claim.
3. Pick one hardened mortgage benchmark/protocol, then synchronize the dataset, builder, docs, outputs, and paper around that single artifact.
4. If the mortgage section stays, downgrade the frontier/fairness rhetoric to what threshold-free AUPRC actually supports, or add a real dev-split operating-point-fair evaluation for that benchmark.
5. Make `make_figures.py` read committed JSON outputs, or change the docs to say the figures are hand-grounded / inline-coded rather than provenance-linked.
6. Pin dataset revisions or snapshot the exact evaluation rows locally for `eval_corrected.py` and `verify_novel.py`; otherwise the “reproducible” and “self-contained” language should be narrowed.

## Resolutions applied

Verified every code-level claim against the files before acting; all were accurate.

- **F1 (notebook ≠ paper artifact; fail-closed GPT).** Reframed `research/README.md` and `research/notebooks/README.md`: the notebook is a **companion demo**, not the paper's reproduction path; the paper's numbers are produced by `scripts/`. Noted the notebook's GPT baseline uses a demo-grade fail-closed default distinct from the paper's abstain policy. *(Did not regenerate the notebook — a ~2h + API job; flagged instead.)*
- **F2 (GPT fix not re-grounded in-house).** Added an explicit reproducibility caveat to the paper's bug list (`sec:bugs`): the released in-house GPT point estimate came from a single cached run whose error branch was fail-closed, so it is bias-free only if no API errors occurred; an abstention-logged re-run is pending (the abstain policy *is* exercised in the mortgage eval). **Not yet re-run** (paid API + could shift the tie) — offered to the author.
- **F3 (hardened-benchmark inconsistency).** Synchronized around the committed **334-row trap-typed** set: rewrote `docs/mortgage-benchmark-hard-results.md`, fixed the `README.md` reproduction chain, and marked `build_hard_jsonl.mjs` / `wf_build_hard_benchmark_v2.mjs` as **superseded** (they built the earlier synthetic 318-row set, not the committed file).
- **F4 (non-matched operating points).** Hedged the mortgage case study: the FPR comparison is explicitly flagged as *not* a matched operating point (local guards at own best-F1, gpt at native), and threshold-free AUPRC is stated as the claim leaned on.
- **F5 (figure provenance).** Corrected `README.md` and `paper/README.md`: `make_figures.py` draws figures from **inline hand-entered** values, not from the JSON.
- **F6 (self-contained / pinned data).** Narrowed the "nothing depends on anything outside `research/`" language to "self-contained in code"; added a Caveats section noting the eval scripts reconstruct rows from unpinned HF datasets and that `outputs/` is gitignored/derived.
- **F7 (mortgage weight).** Downgraded the mortgage case study to **exploratory** in the abstract and contributions (seed-scale, single-annotator), not one of the central results.

Still open (author decision): re-run the in-house GPT baseline with abstention logging (F2); optionally regenerate the companion notebook to the paper's protocol or retire it (F1); pin HF dataset revisions (F6).
