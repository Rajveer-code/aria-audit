#!/usr/bin/env bash
# Reproduce ARIA-Audit headline numbers from scratch.
# Tested on RTX 4060 8GB, Ubuntu 22.04 / Windows 11 WSL.
#
# Expected total runtime (seed suites, 10 items each): ~8 minutes on RTX 4060 8 GB.
# Expected total runtime (full suites, --full flag):   ~3.5 hours on RTX 4060 8 GB.
#
# Usage:
#   bash reproduce.sh          # seed suites (10 items) — smoke test
#   bash reproduce.sh --full   # full suites (300/300/400/200 items) — publication numbers
set -euo pipefail

FULL_EVAL=false
if [[ "${1:-}" == "--full" ]]; then
  FULL_EVAL=true
fi

echo "==> Step 1: install ARIA-Audit"
pip install -e .[dev]

echo "==> Step 2: pull Ollama models (one-time, ~6 GB)"
ollama pull qwen3:8b-q4_K_M
ollama pull granite-guardian:3.2-2b || echo "WARN: granite-guardian:3.2-2b not yet on Ollama, fall back to HuggingFace path"

echo "==> Step 3: VRAM gate test"
# Verifies the full model stack fits within the 7.6 GB VRAM ceiling on an RTX 4060 8 GB.
# Exits non-zero if peak allocation exceeds the ceiling; blocks subsequent steps.
aria-audit vram-gate \
  --model qwen3:8b-q4_K_M \
  --aux-models hhem2.1 bge-m3 phi4-mini \
  --ceiling-gb 7.6 \
  --out eval/results/vram_gate.json
echo "    VRAM gate passed — see eval/results/vram_gate.json"

echo "==> Step 4: run eval over BBQ + BOLD + CPFE suites"
if [[ "$FULL_EVAL" == "true" ]]; then
  BBQ_SUITE=eval/suites/bbq_subset_300.jsonl
  BOLD_SUITE=eval/suites/bold_subset_300.jsonl
  CPFE_SUITE=eval/suites/cpfe_custom_400.jsonl
  CF_SUITE=eval/suites/counterfactual_pairs_200.jsonl
  echo "    Full eval mode: 300 + 300 + 400 + 200 items"
else
  BBQ_SUITE=eval/suites/bbq_subset.jsonl
  BOLD_SUITE=eval/suites/bold_subset.jsonl
  CPFE_SUITE=eval/suites/cpfe_custom.jsonl
  CF_SUITE=eval/suites/counterfactual_pairs.jsonl
  echo "    Seed eval mode: 10 + 10 + 10 + 10 items (smoke test only)"
fi

aria-audit run --model qwen3:8b-q4_K_M --eval "$BBQ_SUITE"  --out eval/results/bbq
aria-audit run --model qwen3:8b-q4_K_M --eval "$BOLD_SUITE" --out eval/results/bold
aria-audit run --model qwen3:8b-q4_K_M --eval "$CPFE_SUITE" --out eval/results/cpfe
aria-audit run --model qwen3:8b-q4_K_M --eval "$CF_SUITE"   --out eval/results/counterfactual \
  --axes equity attribution

echo "==> Step 5: compare baselines"
# Runs Granite Guardian, RAGAS, and LlamaGuard over the same eval suites and
# collects per-axis metrics into a single CSV for the headline table.
python -m eval.compare_baselines \
  --aria-results eval/results/ \
  --baselines granite_guardian ragas llamaguard \
  --out eval/results/headline_table.csv
echo "    Baseline comparison written to eval/results/headline_table.csv"

echo "==> Step 6: generate figures"
python -m paper.figs.render_all \
  --results eval/results/headline_table.csv \
  --out paper/figs/rendered/
echo "    Figures written to paper/figs/rendered/"

echo "==> Step 7: verify eval/results/ artifacts"
if [[ ! -f eval/results/headline_table.csv ]]; then
  echo "ERROR: eval/results/headline_table.csv not found — Step 5 may have failed."
  exit 1
fi
# Check for minimum expected columns in the CSV header
EXPECTED_COLS="axis,aria,granite_guardian,ragas,llamaguard"
HEADER=$(head -1 eval/results/headline_table.csv | tr '[:upper:]' '[:lower:]')
for col in axis aria ragas llamaguard; do
  if [[ "$HEADER" != *"$col"* ]]; then
    echo "ERROR: headline_table.csv missing expected column: $col"
    exit 1
  fi
done
echo "    eval/results/headline_table.csv verified."

echo ""
echo "==> Done. Artifacts:"
echo "    eval/results/headline_table.csv  — headline numbers"
echo "    eval/results/vram_gate.json      — VRAM allocation log"
echo "    paper/figs/rendered/             — publication figures"
echo ""
if [[ "$FULL_EVAL" == "false" ]]; then
  echo "NOTE: ran on seed suites (10 items each). Re-run with --full for publication numbers."
  echo "      Expected full-suite runtime: ~3.5 hours on RTX 4060 8 GB."
else
  echo "NOTE: full-suite run complete. Expected total runtime was ~3.5 hours on RTX 4060 8 GB."
fi
