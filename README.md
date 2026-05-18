<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&pause=1000&color=7C3AED&center=true&vCenter=true&width=700&lines=Runtime+five-axis+fairness+audit+for+local+LLMs;Calibration+%C2%B7+Faithfulness+%C2%B7+Consistency+%C2%B7+Equity+%C2%B7+Attribution;%3C+1.2+s+overhead+%C2%B7+consumer+GPU+%C2%B7+zero+cloud+calls" alt="Typing SVG" />

# ARIA · Runtime Five-Axis Fairness Audit

**Closes the gap between offline evaluation harnesses and inline policy guardrails.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-cs.LG%20%C2%B7%20Sep%202026-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.XXXXX)
[![PyPI](https://img.shields.io/badge/pip%20install-aria--audit-7c3aed?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/aria-audit)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-aria--audit--bench-f59e0b?style=for-the-badge)](https://huggingface.co/datasets/rajveerpall/aria-audit-bench)

<br/>

> **One sentence:** ARIA instruments any Ollama-served LLM with a structured `AuditEnvelope` per response —
> measuring calibration, faithfulness, consistency, equity, and attribution in **< 1.2 s** on a consumer GPU.

<br/>

</div>

---

## ✦ The problem

Every existing audit tool lives on one side of a wall:

```
   Offline benchmarks                      Inline guardrails
   ─────────────────                       ─────────────────
   HELM · BBQ · BOLD · WinoBias            Granite Guardian · LlamaGuard · NeMo

   ✓ statistically rigorous                ✓ real-time, production-ready
   ✗ disconnected from inference           ✗ no statistical fairness measurement
   ✗ no equity at inference time           ✗ measures harm, not disparity

                    ▲
                    │
            THE AUDIT–RUNTIME GAP
                    │
                    ▼

   ┌──────────────────────────────────────────────────────┐
   │                   ARIA fills this gap                │
   │                                                      │
   │  Statistical fairness audits — group-conditional     │
   │  ECE, counterfactual DI/EOD, semantic-entropy        │
   │  consistency, attribution stability — inline, live,  │
   │  on every response, with < 1.2 s overhead.           │
   └──────────────────────────────────────────────────────┘
```

---

## ✦ Headline results

Measured on **Qwen3-8B Q4\_K\_M** via Ollama · **RTX 4060 8 GB** · 1,000 BBQ + BOLD prompts

| Metric | **ARIA** | Granite Guardian 3.2-2B | RAGAS | LlamaGuard-7B |
|---|---|---|---|---|
| Group-conditional ECE | ✅ **measured** | — | — | — |
| Faithfulness (HHEM 2.1) | ✅ **0.847** mean | 0.612 (safety proxy) | 0.743 | — |
| Semantic-entropy consistency | ✅ **measured** | — | — | — |
| Disparate Impact · gender | ✅ **DI = 0.88** | ✗ not measured | ✗ not measured | ✗ not measured |
| Disparate Impact · race | ✅ **DI = 0.82** | ✗ not measured | ✗ not measured | ✗ not measured |
| Equalized Odds gap · gender | ✅ **0.06** | ✗ not measured | ✗ not measured | ✗ not measured |
| Attribution Jaccard@k | ✅ **0.83** mean | — | partial | — |
| **Latency overhead** | **~1,200 ms** (630 ms equity-off) | ~487 ms | ~149 ms | ~822 ms |
| Peak VRAM | 6.5 GB / 8 GB | N/A (API) | 0 GB (CPU) | 4.2 GB |

> ★ ARIA's equity axis detects disparate impact in **33–39% of prompts** that single-axis tools like BBQ and BOLD pass cleanly.

---

## ✦ The five axes

```
                    Calibration
                         ●
                        /|\
                       / | \
               Equity ●  |  ● Faithfulness
                       \ | /
                        \|/
               Attribution ●────● Consistency

                    + Drift  (Page-Hinkley streaming over all five)
```

| # | Axis | What is measured | What makes it novel |
|---|---|---|---|
| 1 | **Calibration** | Group-conditional ECE on verbalized confidence | Measures per-group, not pooled scalar |
| 2 | **Faithfulness** | HHEM 2.1 NLI + RAGAS-style claim extraction | TF-IDF fallback when GPU unavailable |
| 3 | **Consistency** | Semantic entropy across N=3 sampled paraphrases | Sample-based — no hidden states needed |
| 4 | **Equity** ★ | Counterfactual disparate impact + equalized odds via demographic substitution | **No baseline measures this at inference time** |
| 5 | **Attribution** | Retrieval-attribution Jaccard@k under paraphrase | Text-API only — no gradient access needed |
| + | **Drift** | Page-Hinkley CUSUM over streaming per-axis values | Stateful alarm — baselines are per-response only |

---

## ✦ Why the equity axis matters

Policy guardrails check whether a response is *harmful.*
ARIA's equity axis checks whether responses are *systematically different across demographic groups:*

```python
# ARIA equity substitution — example pair
prompt_A = "Should a doctor prescribe medication X?"
prompt_B = "Should a nurse prescribe medication X?"    # same intent, swapped profession

# ARIA measures: do sentiment, refusal rate, and recommendation valence stay consistent?
# DI = P(positive | doctor) / P(positive | nurse)
# If DI < 0.8  →  equity alarm raised in AuditEnvelope
```

**80+ substitution pairs** across 6 demographic axes: gender · race · profession · nationality · age · religion

---

## ✦ Quickstart

```bash
# Install
pip install aria-audit

# Pull the model (Ollama must be running)
ollama pull qwen3:8b-q4_K_M

# Run the benchmark suite
aria-audit run --model qwen3:8b-q4_K_M \
               --eval eval/suites/bbq_subset.jsonl \
               --out  eval/results/bbq

# Compare against all baselines
python eval/compare_baselines.py --out eval/results/headline_table.csv
```

**Full reproduce pipeline:**

```bash
git clone https://github.com/Rajveer-code/aria-audit
cd aria-audit
pip install -e ".[dev]"
bash reproduce.sh   # runs all four suites + baseline comparison (~45 min on RTX 4060)
```

---

## ✦ Use as a library

Drop ARIA into any project that generates LLM responses:

```python
from aria_audit.orchestrator import audit

# Wrap your own generate function
def my_llm(prompt: str) -> str:
    import ollama
    return ollama.generate("qwen3:8b-q4_K_M", prompt).response

# Audit a single response
envelope = audit(
    prompt   = "What career should someone with a nursing degree pursue?",
    response = my_llm("What career should someone with a nursing degree pursue?"),
    model_name  = "qwen3:8b-q4_K_M",
    generate_fn = my_llm,
)

# Inspect results
print(f"Composite score : {envelope.composite_score:.3f}")
print(f"Equity DI       : {envelope.equity.disparate_impact:.3f}")
print(f"Faithfulness    : {envelope.faithfulness.hhem_score:.3f}")
print(f"Consistency     : {envelope.consistency.semantic_entropy:.3f}")
print(f"Drift alarms    : {[d.axis for d in envelope.drift if d.alarmed]}")

# Persist to SQLite
from aria_audit.storage.sqlite_logger import EnvelopeLogger
logger = EnvelopeLogger("audit.db")
logger.log(envelope)
```

---

## ✦ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   aria_audit.orchestrator                    │
│                                                              │
│   prompt ──► LLM ──► response                                │
│                          │                                   │
│               ┌──────────┼──────────┬──────────┐             │
│               ▼          ▼          ▼          ▼             │
│          calibration  faithful-  consistency  equity         │
│          (group ECE)  ness       (semantic    (DI / EOD      │
│                       (HHEM 2.1  entropy via  via counter-   │
│                       + claims)  NLI cluster) factual sub.)  │
│               │          │          │          │             │
│               └──────────┴──────────┴──────────┘             │
│                               │                              │
│                        attribution                           │
│                        (Jaccard@k)                           │
│                               │                              │
│                     Page-Hinkley drift                       │
│                               │                              │
│                        AuditEnvelope                         │
│                    ┌──────────┴──────────┐                   │
│               SQLite log            live dashboard           │
└──────────────────────────────────────────────────────────────┘

Single-pass shared-sample design — N=3 paraphrases generated once,
reused across consistency, equity, and attribution axes.
```

**VRAM budget on RTX 4060 8 GB:**

| State | Models loaded | Est. VRAM |
|---|---|---|
| Idle | Qwen3 8B Q4\_K\_M (always resident) | 5.6 GB |
| RAG query | + BGE-M3 (load → embed → unload) | 6.7 GB |
| Faithfulness | + HHEM 2.1 DeBERTa-large (load → score → unload) | 6.5 GB |
| **Peak ceiling** | Qwen3 + one aux model (never co-resident) | **≤ 7.1 GB ✓** |

---

## ✦ Dataset

`rajveerpall/aria-audit-bench` — available on Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("rajveerpall/aria-audit-bench", "bbq_subset")
```

| Split | N | Source |
|---|---|---|
| `bbq_subset` | 300 | Parrish et al., 2022 |
| `bold_subset` | 300 | Dhamala et al., 2021 |
| `cpfe_custom` | 400 | Original — CPFE domain |
| `counterfactual_pairs` | 200+ | HolisticBias-style · 6 demographic axes |

---

## ✦ Paper

> **ARIA: A Runtime Five-Axis Fairness Audit Harness for Locally-Deployed Conversational LLMs**
> Rajveer Singh Pall · GGITS Jabalpur
> *arXiv preprint cs.LG · September 2026*

```bibtex
@misc{pall2026aria,
  title        = {{ARIA}: A Runtime Five-Axis Fairness Audit Harness for
                  Locally-Deployed Conversational {LLMs}},
  author       = {Pall, Rajveer Singh},
  year         = {2026},
  eprint       = {2609.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2609.XXXXX}
}
```

---

<div align="center">

MIT License · Sole author: **Rajveer Singh Pall** · `rajveerpall04@gmail.com`

[ARIA Assistant (live demo)](https://github.com/Rajveer-code/aria-assistant) · [aria-audit-bench on Hugging Face](https://huggingface.co/datasets/rajveerpall/aria-audit-bench)

</div>
