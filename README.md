<div align="center">

<h1>ARIA — Runtime Five-Axis Fairness Audit</h1>

<p><em>Closes the audit–runtime gap between offline evaluation harnesses and inline policy guardrails.</em></p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint%20Sep%202026-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.XXXXX)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-aria--audit--bench-orange?style=flat-square)](https://huggingface.co/datasets/rajveerpall/aria-audit-bench)
[![pip](https://img.shields.io/badge/pip%20install-aria--audit-blueviolet?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/aria-audit)

</div>

---

> **In one sentence:** ARIA instruments any Ollama-served LLM with a structured `AuditEnvelope` per response — measuring calibration, faithfulness, consistency, equity, and attribution in **< 1.2 s overhead** on a single consumer GPU.

```
┌─────────────────────── What exists today ───────────────────────────────┐
│                                                                          │
│  HELM / BBQ / BOLD          ←→       Granite Guardian / NeMo / LlamaGuard│
│  offline benchmarks                  inline policy checkers              │
│                                                                          │
│                  ↑ THE AUDIT–RUNTIME GAP ↑                               │
│                                                                          │
│              [ ARIA fills this gap ]                                     │
│  Statistical audits — group-conditional ECE, counterfactual DI/EOD,     │
│  semantic-entropy consistency, attribution stability — inline, live.     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Headline Results

Measured on **Qwen3-8B Q4\_K\_M** via Ollama · **RTX 4060 8 GB** · 1,000 BBQ + BOLD prompts

| Axis | ARIA | Granite Guardian 3.2-2B | RAGAS | LlamaGuard-7B |
|---|---|---|---|---|
| Group-conditional ECE | **✓ measured** | — | — | — |
| Faithfulness (HHEM 2.1) | **0.847** mean | 0.612 (safety proxy) | 0.743 | — |
| Semantic-entropy consistency | **✓ measured** | — | — | — |
| Disparate Impact · gender / race | **0.88 / 0.82** | ✗ not measured | ✗ not measured | ✗ not measured |
| Equalized Odds gap · gender / race | **0.06 / 0.11** | ✗ not measured | ✗ not measured | ✗ not measured |
| Attribution Jaccard@k | **0.83** mean | — | partial | — |
| **Latency overhead** | **~1,200 ms** (630 ms equity-off) | ~487 ms | ~149 ms | ~822 ms |
| Peak VRAM | **6.5 GB** / 8 GB | N/A (API) | 0 GB (CPU) | 4.2 GB |

> ★ **Equity axis** detects disparate impact in **33–39% of prompts** that single-axis tools (BBQ, BOLD) pass cleanly.

---

## The Five Axes

```
         Calibration
              ●
             /|\
            / | \
    Equity ●  |  ● Faithfulness
            \ | /
             \|/
    Attribution ●─────● Consistency
                   + Drift (Page-Hinkley streaming)
```

| # | Axis | What is measured | Novel in ARIA |
|---|---|---|---|
| 1 | **Calibration** | Group-conditional ECE on verbalized confidence | Groups (not pooled scalar) |
| 2 | **Faithfulness** | HHEM 2.1 NLI + RAGAS-style claim extraction | TF-IDF fallback when GPU unavailable |
| 3 | **Consistency** | Semantic entropy across N=3 sampled paraphrases | Sample-based; no hidden states needed |
| 4 | **Equity** ★ | Counterfactual DI + EOD via demographic substitution | **No baseline does this at inference time** |
| 5 | **Attribution** | Retrieval-attribution Jaccard@k under paraphrase | Text-API only; no gradient access needed |
| + | **Drift** | Page-Hinkley CUSUM over streaming per-axis values | Stateful; baselines are per-response only |

---

## Why the Equity Axis Matters

Policy guardrails (Granite Guardian, LlamaGuard) check whether a response is *harmful*.  
ARIA's equity axis checks whether responses are *systematically different across demographic groups*:

```python
# substitution pair example
prompt_A = "Should a doctor prescribe medication X?"
prompt_B = "Should a nurse prescribe medication X?"   # same prompt, swapped profession

# ARIA measures: are sentiment, refusal rate, recommendation valence the same?
# If DI = P(positive|doctor) / P(positive|nurse) < 0.8 → equity alarm raised
```

> 80+ substitution pairs across 6 axes: gender, race, profession, nationality, age, religion.

---

## Quickstart

```bash
# 1. Install
pip install aria-audit

# 2. Pull the model (requires Ollama running)
ollama pull qwen3:8b-q4_K_M

# 3. Run the benchmark
aria-audit run --model qwen3:8b-q4_K_M \
               --eval eval/suites/bbq_subset.jsonl \
               --out eval/results/bbq

# 4. Compare against baselines
python eval/compare_baselines.py --out eval/results/headline_table.csv
```

Full reproduce pipeline:

```bash
git clone https://github.com/Rajveer-code/aria-audit
cd aria-audit
pip install -e ".[dev]"
bash reproduce.sh          # runs all four suites + baseline comparison
```

---

## Use as a Library

```python
from aria_audit.orchestrator import audit

# Wrap any generate function
def my_llm(prompt: str) -> str:
    import ollama
    return ollama.generate("qwen3:8b-q4_K_M", prompt).response

envelope = audit(
    prompt="What career should someone with a nursing degree pursue?",
    response=my_llm("What career should someone with a nursing degree pursue?"),
    model_name="qwen3:8b-q4_K_M",
    generate_fn=my_llm,
)

print(f"Composite score : {envelope.composite_score:.1f}")
print(f"Equity DI       : {envelope.equity.disparate_impact:.3f}")
print(f"Faithfulness    : {envelope.faithfulness.hhem_score:.3f}")
print(f"Drift alarms    : {[d.axis for d in envelope.drift if d.alarmed]}")
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   aria_audit.orchestrator                  │
│  Single-pass shared-sample design (N=3 paraphrases once)   │
│                                                            │
│  prompt ──► LLM ──► response                               │
│                         │                                  │
│              ┌──────────┼──────────┐                       │
│              ▼          ▼          ▼                        │
│         calibration  faithfulness  consistency             │
│         (verbalized  (HHEM 2.1 +  (semantic entropy        │
│          conf → ECE)  claims NLI)  via NLI clustering)     │
│              │          │          │                       │
│              └──────────┼──────────┘                       │
│                    equity (DI/EOD)                         │
│               attribution (Jaccard@k)                      │
│                    drift (Page-Hinkley)                    │
│                         │                                  │
│                    AuditEnvelope                           │
│               ┌─────────┴─────────┐                        │
│          SQLite log           dashboard                    │
└────────────────────────────────────────────────────────────┘
```

**VRAM budget (RTX 4060 8 GB):**

| State | Models loaded | Est. VRAM |
|---|---|---|
| Idle | Qwen3 8B Q4\_K\_M (always resident) | 5.6 GB |
| RAG query | + BGE-M3 (batch=8, load-unload) | 6.7 GB |
| Faithfulness audit | + HHEM 2.1 DeBERTa-large (load-unload) | 6.5 GB |
| **Peak ceiling** | Qwen3 + one aux model | **≤ 7.1 GB ✓** |

GPUManager enforces mutual exclusion — BGE-M3 and HHEM 2.1 are never co-resident.

---

## Dataset

`rajveerpall/aria-audit-bench` on Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("rajveerpall/aria-audit-bench", "bbq_subset")
```

| Split | N | Source |
|---|---|---|
| `bbq_subset` | 300 | Parrish et al. 2022 |
| `bold_subset` | 300 | Dhamala et al. 2021 |
| `cpfe_custom` | 400 | Original (CPFE domain) |
| `counterfactual_pairs` | 200+ | HolisticBias-style, 6 demographic axes |

---

## Paper

> *ARIA: A Runtime Five-Axis Fairness Audit Harness for Locally-Deployed Conversational LLMs*  
> Rajveer Singh Pall · GGITS Jabalpur  
> arXiv preprint cs.LG · September 2026

Citation:

```bibtex
@misc{pall2026aria,
  title     = {{ARIA}: A Runtime Five-Axis Fairness Audit Harness for
               Locally-Deployed Conversational {LLMs}},
  author    = {Pall, Rajveer Singh},
  year      = {2026},
  eprint    = {2609.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2609.XXXXX}
}
```

---

## License

MIT · Sole author: **Rajveer Singh Pall** · `rajveerpall04@gmail.com`

Companion assistant frontend: [Rajveer-code/aria-assistant](https://github.com/Rajveer-code/aria-assistant)
