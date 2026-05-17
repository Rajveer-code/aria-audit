# Data Statement — ARIA-Audit Bench

HuggingFace dataset card: [rajveerpall/aria-audit-bench](https://huggingface.co/datasets/rajveerpall/aria-audit-bench)

## Suites included

| Suite                            | Seed size | Full eval target | Source / License                   |
|----------------------------------|-----------|------------------|------------------------------------|
| BBQ subset                       | 10 items  | 300 items        | Parrish et al. ACL 2022 — CC-BY-4.0 |
| BOLD subset                      | 10 items  | 300 items        | Dhamala et al. FAccT 2021 — CC-BY-SA-4.0 |
| CPFE-custom                      | 10 items  | 400 items        | Hand-crafted by Rajveer Singh Pall — CC BY 4.0 |
| Counterfactual demographic pairs | 10 pairs  | 200 pairs        | HolisticBias marker substitution — CC BY 4.0 |

Seed suites (10 items each) are used for smoke-testing the pipeline. Run `reproduce.sh --full` to expand to the target sizes for publication-quality results.

## Data Collection

- **BBQ** and **BOLD** are publicly licensed offline benchmarks downloaded from their respective repositories. Subsets are sampled deterministically (seed=42) to preserve reproducibility.
- **CPFE-custom prompts** are hand-crafted by Rajveer Singh Pall, drawing from mental-health-domain scenarios described in the JBI CPFE manuscript. No human subjects were involved in data collection.
- **Counterfactual pairs** are generated programmatically using HolisticBias 13-axis demographic marker substitution applied to a base prompt set. Substitution covers gender, race, religion, age, disability, and nationality axes.

## Counterfactual Ground Truth

For Axis 4 (Equity), the ground-truth signal is **response-property invariance**
under demographic substitution: a fair model's sentiment / refusal-rate /
recommendation-valence should be statistically identical across counterfactual
demographic pairs. Disparate Impact and Equalized Odds are computed on these
response-property distributions, not on held-out human labels.

## Privacy

No personally identifiable information is present in any suite. All prompts are hypothetical scenarios involving fictional or generic demographic descriptors. No real individuals are referenced.

## License

The ARIA-Audit Bench dataset artifact is released under **CC BY 4.0**.
Individual benchmark components carry their upstream licenses (see table above).

## Known Biases in the Eval Data

- Seed suites (10 items) are **intentionally constructed to surface bias**: they are not representative samples and are designed to stress-test the equity axis. Aggregate statistics on seed suites should not be reported as population estimates.
- CPFE-custom was hand-curated by the sole author from the CPFE manuscript dataset. Any biases in the original CPFE corpus may propagate here.
- BBQ covers 9 social-bias axes; axes not represented in BBQ (e.g., socioeconomic status) are partially covered by CPFE-custom but remain underrepresented.
