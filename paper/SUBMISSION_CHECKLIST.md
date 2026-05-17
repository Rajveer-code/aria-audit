# arXiv Submission Checklist

## Hard deadline: September 15, 2026

---

## Before submission

- [ ] Replace all placeholder numbers (33–39%, 17-point drop, 1,200 ms, 32% race-axis bias rate, etc.) with real numbers from `eval/results/headline_table.csv` if the full eval has been re-run.
- [ ] Run full eval end-to-end:
  ```bash
  aria-audit run --model qwen3:8b-q4_K_M --eval eval/suites/bbq_subset.jsonl --out eval/results/bbq
  aria-audit run --model qwen3:8b-q4_K_M --eval eval/suites/bold_subset.jsonl --out eval/results/bold
  aria-audit run --model qwen3:8b-q4_K_M --eval eval/suites/cpfe_custom.jsonl --out eval/results/cpfe
  aria-audit run --model qwen3:8b-q4_K_M --eval eval/suites/counterfactual_pairs.jsonl --out eval/results/cf --run-equity
  python eval/compare_baselines.py --out eval/results/headline_table.csv
  ```
- [ ] Regenerate all five figures from real data:
  ```bash
  python paper/figs/gen_fig_coverage_radar.py
  python paper/figs/gen_fig_latency.py
  python paper/figs/gen_fig_equity_distributions.py
  python paper/figs/gen_fig_drift.py
  python paper/figs/gen_fig_ablation.py
  ```
- [ ] Re-compile and confirm zero LaTeX errors:
  ```bash
  cd paper
  pdflatex arxiv_preprint.tex
  bibtex   arxiv_preprint
  pdflatex arxiv_preprint.tex
  pdflatex arxiv_preprint.tex
  ```
- [ ] Final PDF: 8–10 pages of main body (currently 11 with refs — trim if exceeding venue cap).
- [ ] Confirm all `\cite{}` keys resolve in `refs.bib` (no `[?]` markers in PDF).
- [ ] Confirm all `\ref{fig:...}` and `\ref{tab:...}` resolve (no `??` markers).
- [ ] Spell-check pass: `aspell -c arxiv_preprint.tex` (skip LaTeX commands).
- [ ] Upload dataset to Hugging Face: `rajveerpall/aria-audit-bench`
  - JSONL files: bbq_subset, bold_subset, cpfe_custom, counterfactual_pairs
  - Dataset card with DATA_STATEMENT.md content
- [ ] Push `aria-audit/` to GitHub as a public repo (`https://github.com/Rajveer-code/aria-audit`).
- [ ] Verify `pip install aria-audit` works from a clean conda env.
- [ ] Update repo README with the headline result + reproduce script + arXiv URL (after submission).

---

## arXiv submission

- Go to: <https://arxiv.org/submit>
- **Primary category:** `cs.LG` (Machine Learning)
- **Cross-list:** `cs.CY` (Computers and Society — covers fairness)
- **License:** `CC BY 4.0`
- **Upload bundle:** zip containing
  - `arxiv_preprint.tex`
  - `refs.bib`
  - `neurips_2024.sty`
  - `figs/` directory (all five PDFs)
  - Optionally `arxiv_preprint.bbl` if you do not want arXiv to re-run bibtex
- **Title:** *ARIA: A Runtime Five-Axis Fairness Audit Harness for Locally-Deployed Conversational LLMs*
- **Abstract:** copy verbatim from the LaTeX source (already under 250 words).
- **Authors:** Rajveer Singh Pall (sole author), GGITS Jabalpur, `rajveerpall04@gmail.com`
- **Comments box:** `10 pages, 5 figures. Code: github.com/Rajveer-code/aria-audit  Dataset: huggingface.co/datasets/rajveerpall/aria-audit-bench`

---

## After arXiv goes live

- [ ] Copy the arXiv URL (format: `https://arxiv.org/abs/2609.XXXXX`).
- [ ] Add the URL to the abstract footnote in the LaTeX source for the v2 update.
- [ ] Reference the URL in MSc SOPs (ETH Zurich, TUM, EPFL, Edinburgh).
- [ ] Tweet announcement linking to the HF dataset card.
- [ ] Cross-post a 1,500-word negative-findings note on LessWrong / Alignment Forum.
- [ ] Submit to **NeurIPS 2026 SoLaR workshop** — check deadline at <https://solar-neurips.github.io>.

---

## NeurIPS SoLaR workshop format (condensed version)

- **Max:** 5 pages + unlimited references.
- **Style:** Download the NeurIPS 2026 workshop style file when released.
- **Cuts from arXiv version:**
  - Drop Sections 3.5 (Axis 5 details) and 3.6 (drift) — summarise in one paragraph each.
  - Compress Sections 5.2 (latency) and 5.3 (equity) into a single results subsection.
  - Drop the Why-not-chain-GG-RAGAS paragraph (move to appendix).
- **Workshop pitch:** lead with the equity axis and the 33–39% unique-detection rate.
  Frame around societal impact: an inline fairness check that runs at deployment time on a single consumer GPU lowers the cost of fairness monitoring by orders of magnitude.

---

## FAccT 2027 main track (later)

- **Deadline:** typically October–November 2026 for ACM FAccT 2027.
- **Format:** ACM SIG Conference proceedings, 12 pages + refs.
- **Cuts/Expansions vs. arXiv:**
  - Add a dedicated Ethics section (DATA_STATEMENT.md content as a paper section).
  - Expand the substitution vocabulary discussion: how the 80+ pairs were curated, who reviewed them, what was excluded and why.
  - Add a multi-turn extension experiment if scoped in time.
  - Position against the broader algorithmic-fairness literature, not just LLM tooling.

---

## Compile command (one-liner)

```bash
cd paper && pdflatex arxiv_preprint.tex && bibtex arxiv_preprint && pdflatex arxiv_preprint.tex && pdflatex arxiv_preprint.tex
```

Current compile status (as of 2026-05-17):
- ✓ Zero LaTeX errors
- ✓ Zero undefined references
- ✓ All five figures render
- ✓ 11 pages including bibliography (10 pages main body + 1 page refs)
- Only warnings: two harmless hyperref `Token not allowed in a PDF string` warnings from `$\bigstar$` in the table caption — does not affect output PDF.
