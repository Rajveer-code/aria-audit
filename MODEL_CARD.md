# Model Card — ARIA-Audit Layer

## Audit Layer Overview
- **What:** A model-agnostic runtime audit instrumentation layer.
- **Not a model itself.** Wraps any LLM (primary target: Qwen3-8B Q4_K_M via Ollama, always-resident anchor).
- **Five axes emit one `AuditEnvelope` per response.**

## Intended Use
- Personal assistant augmentation and research demonstration for fairness/calibration/faithfulness/equity trade-offs in locally-deployed LLMs.
- Educational analysis of audit-axis trade-offs on single-user conversational deployments.
- NOT intended for production multi-user serving environments.

## Out-of-Scope Uses
- Medical diagnosis or clinical decision support.
- Legal advice or legal document interpretation.
- High-stakes financial or safety-critical decisions.
- Any context where the audit output would substitute for human expert judgment.

## Components & External Models

| Component       | Source                                      | License        | VRAM (GB) |
|-----------------|---------------------------------------------|----------------|-----------|
| Qwen3 8B Q4_K_M | Ollama / HuggingFace (always-resident anchor) | Apache 2.0   | 5.6       |
| BGE-M3          | BAAI/bge-m3                                 | MIT            | 1.1 (bs=8)|
| HHEM 2.1        | vectara/hallucination_evaluation_model      | Apache 2.0     | 0.9       |
| Phi-4-mini (CPU)| microsoft/Phi-4-mini-instruct (Q4 GGUF)     | MIT            | 0 (RAM)   |
| Granite Guardian 3.2-2B (baseline) | ibm-granite/granite-guardian | Apache 2.0 | 1.6 |

**Auxiliary models:** HHEM 2.1 (`vectara/hallucination_evaluation_model`) for faithfulness scoring, BGE-M3 for retrieval-attribution, Phi-4-mini running CPU-only for lightweight classification subtasks.

## Hardware Requirements
- **Minimum:** 8 GB VRAM GPU.
- **Tested on:** RTX 4060 8 GB (peak allocation 6.5 GB; headroom ceiling 7.6 GB).
- System RAM: 16 GB recommended (Phi-4-mini CPU path uses ~4 GB RAM).
- Ollama must be running and `qwen3:8b-q4_K_M` must be pulled before audit start.

## Known Limitations
- **Equity axis** assumes response-level sentiment/refusal proxy as the outcome variable; no access to held-out human labels for DI/EOD ground truth.
- **Attribution** is Jaccard-based (retrieval overlap under paraphrase), not gradient-based; does not capture token-level saliency.
- **Calibration** requires verbalized confidence in the model output; models that refuse to state confidence yield null ECE.
- **Lookback-Lens** dropped — Ollama API exposes no attention weights.
- **Captum integrated-gradients** dropped — Ollama exposes no gradients; attribution reframed as retrieval-attribution Jaccard.
- **Semantic-Entropy-Probes** dropped — needs hidden states; consistency falls back to sample-based Kuhn 2023.
- DI/EOD ground truth is **counterfactual demographic substitution**, not held-out labels — see DATA_STATEMENT.md.

## Evaluation
See `eval/results/headline_table.csv` after running `reproduce.sh`. Full methodology in the preprint [coming Sep 2026].
