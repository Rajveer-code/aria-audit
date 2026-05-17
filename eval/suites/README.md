# Eval Suites

JSONL files, one item per line. Schema:

```json
{
  "id": "bbq_0042",
  "prompt": "...",
  "reference_answer": "...",
  "group_tag": "ambiguous_gender",
  "axes_relevant": ["equity", "calibration"]
}
```

Files (Phase 0 deliverable):
- `bbq_subset.jsonl` — ~300 items from Parrish 2022 BBQ
- `bold_subset.jsonl` — ~300 items from BOLD FAccT 2021
- `cpfe_custom.jsonl` — ~400 hand-curated CPFE mental-health prompts
- `counterfactual_pairs.jsonl` — ~200 HolisticBias-style demographic substitutions
