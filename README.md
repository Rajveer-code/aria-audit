# ARIA-Audit

**Runtime five-axis fairness audit for locally-deployed conversational LLMs.**
Operationalizes the CPFE (Cross-Platform Fairness Evaluation) framework on
deployed inference, closing the audit-runtime gap between offline evaluation
harnesses (HELM, RAGAS) and inline policy guardrails (Granite Guardian, NeMo).

---

## Install & run

```bash
pip install aria-audit && aria-audit benchmark --model qwen3:8b-q4_K_M
```

Dataset: [rajveerpall/aria-audit-bench](https://huggingface.co/datasets/rajveerpall/aria-audit-bench)
Preprint: [preprint coming Sep 2026]

---

## Headline results (RTX 4060 8 GB, Qwen3-8B Q4_K_M)

| Axis                              | ARIA                         | Granite Guardian 3.2-2B | RAGAS          | LlamaGuard-7B  |
|-----------------------------------|------------------------------|-------------------------|----------------|----------------|
| Group-conditional ECE             | measured                     | —                       | —              | —              |
| Faithfulness (HHEM-aligned)       | 0.847 mean                   | 0.612 (safety proxy)    | 0.743          | —              |
| Semantic-entropy consistency      | measured                     | —                       | —              | —              |
| Disparate Impact (gender / race)  | 0.88 / 0.82 mean             | NOT MEASURED            | NOT MEASURED   | NOT MEASURED   |
| Equalized Odds gap (gender / race)| 0.06 / 0.11                  | NOT MEASURED            | NOT MEASURED   | NOT MEASURED   |
| Retrieval-attribution Jaccard@k   | 0.83 mean                    | —                       | partial        | —              |
| Latency overhead (ms/resp.)       | ~1200 ms total (~630 equity off) | ~487 ms             | ~149 ms        | ~822 ms        |
| Peak VRAM (RTX 4060 8 GB)         | 6.5 GB (ceiling: 7.6 GB)     | N/A (API)               | 0 GB (CPU)     | 4.2 GB         |

## What ARIA covers that baselines don't

- **Group-conditional ECE** — calibration is measured separately per demographic group, not pooled; baselines report only aggregate or omit calibration entirely.
- **Counterfactual Disparate Impact and Equalized Odds** — equity axes are computed from paired demographic-substitution responses; no baseline audits DI or EOD at inference time.
- **Streaming drift detection** — Page-Hinkley over per-axis time series flags distributional shift mid-session; baselines are stateless per-response checkers.

---

## Why it exists

- HELM, BBQ, BOLD evaluate models on **fixed offline benchmarks**.
- Guardrails AI / NeMo / Granite Guardian enforce **policy** at runtime.
- Nobody runs **CPFE-style statistical audits** — group-conditional ECE, counterfactual DI/EOD, semantic-entropy consistency, retrieval-attribution stability — **inline on conversational outputs**.

ARIA-Audit is a portable, model-agnostic instrumentation layer that emits a
structured `AuditEnvelope` per response, suitable for SQLite logging, dashboard
display, drift detection, and post-hoc analysis.

## Quickstart

```bash
git clone https://github.com/Rajveer-code/aria-audit
cd aria-audit
pip install -e .[dev]

# Requires Ollama with qwen3:8b-q4_K_M pulled
aria-audit benchmark --model qwen3:8b-q4_K_M --samples 100
```

## Five axes (CPFE-grounded)

| # | Axis                         | Implementation                                                         | Citation                  |
|---|------------------------------|------------------------------------------------------------------------|---------------------------|
| 1 | Calibration                  | Group-conditional ECE on verbalized confidence                         | Tian 2023; QA-Calib ICLR'25 |
| 2 | Faithfulness                 | HHEM 2.1 NLI + RAGAS-style claim extraction                            | Vectara HHEM 2.1          |
| 3 | Consistency                  | Semantic entropy via N=3 sampled paraphrases + bi-NLI clustering       | Kuhn 2023                 |
| 4 | **Equity** (novelty)         | Counterfactual demographic substitution → DI + EOD                     | CPFE 2026; HolisticBias   |
| 5 | Attribution                  | Retrieval-attribution Jaccard@k under paraphrase                       | FASS; CPFE reframe        |
|   | Streaming drift              | Page-Hinkley over per-axis time series                                 | Page 1954                 |

## Citing

```bibtex
@misc{pall2026aria,
  title={ARIA: A Runtime Five-Axis Fairness Audit for Locally-Deployed Conversational LLMs},
  author={Pall, Rajveer Singh},
  year={2026},
  note={Preprint in preparation}
}
```

## License

MIT — see `LICENSE`. Sole author: Rajveer Singh Pall.
