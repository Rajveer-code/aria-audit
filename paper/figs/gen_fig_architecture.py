"""Figure 6 — ARIA system architecture diagram (Gemini image generation).

Requires: GEMINI_API_KEY env var set.
Get key at: https://aistudio.google.com/apikey

Run: python gen_fig_architecture.py
Outputs: fig_architecture_attempt{1,2,3}.png — review and pick best.
"""

import os
import sys
import time

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable.")
    print("  Get a key at: https://aistudio.google.com/apikey")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

MODEL = "gemini-3-pro-image-preview"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
client = genai.Client(api_key=API_KEY)

PROMPT = """
Create a MODERN MINIMAL-style system architecture diagram for an ML conference paper (NeurIPS/FAccT).
The system is called ARIA — a runtime five-axis fairness audit harness for locally-deployed LLMs.

VISUAL STYLE — MODERN MINIMAL:
- Ultra-clean geometric shapes with crisp edges
- Bold color blocks as section backgrounds: slate blue (#E8EDF2) for input/retrieval,
  warm sand (#F5F0E8) for LLM core, cool mint (#E8F2EE) for audit pipeline, lavender (#EEE8F5) for output
- Component boxes: WHITE fill, NO visible border, subtle drop shadow (1px, 4px blur)
- Rounded corners (12px radius) on all boxes
- ONE accent color per section: Deep blue (#2563EB) for retrieval, Amber (#D97706) for LLM,
  Emerald (#059669) for audit, Violet (#7C3AED) for storage/output
- Arrows: thin (1.5px), dark gray (#6B7280), small filled circle at source, clean arrowhead at target
- Typography: bold 600 weight for section headers, regular 400 for component labels
- Labels INSIDE boxes
- Generous whitespace (24px between elements)
- NO decorative icons — let the structure speak
- Background: pure white

COLOR PALETTE:
- Input region background: #E8EDF2 (slate blue)
- LLM region background: #F5F0E8 (warm sand)
- Audit region background: #E8F2EE (cool mint)
- Output region background: #EEE8F5 (lavender)
- Accent/arrow colors: #2563EB (blue), #D97706 (amber), #059669 (emerald), #7C3AED (violet)
- Text: #1F2937 (dark gray)
- Arrows: #6B7280 (medium gray)

LAYOUT (left-to-right, three main columns):

COLUMN 1 — INPUT (background: #E8EDF2, accent: #2563EB):
Section header: "INPUT LAYER" (small caps, letter-spaced)
Boxes (stacked vertically):
  [Voice Query] — via faster-whisper distil-large-v3
  [Text Prompt] — direct API input
  [Retrieval Context] — BGE-M3 hybrid retriever → Qdrant → BGE-reranker-v2-m3

COLUMN 2 — LLM CORE (background: #F5F0E8, accent: #D97706):
Section header: "LLM CORE"
Boxes:
  [Qwen3 8B Q4_K_M] — Always-resident anchor (~5.6 GB VRAM)
  Below it: [GPU Manager] — enforces VRAM mutual-exclusion contract
  Small annotation on GPU Manager: "peak ceiling 7.6 GB"
  [Phi-4-mini CPU] — smaller box, to the right, labeled "co-pilot (CPU-only)"
  Arrows: Voice Query → Qwen3 8B, Text Prompt → Qwen3 8B, Retrieval Context → Qwen3 8B
  Arrow: Qwen3 8B → [Response] (right side, going to Column 3)

COLUMN 3 — AUDIT PIPELINE (background: #E8F2EE, accent: #059669):
Section header: "FIVE-AXIS AUDIT"
Boxes arranged in a vertical stack (each is a distinct component):
  [Axis 1: Calibration] — Group-conditional ECE
  [Axis 2: Faithfulness] — HHEM 2.1 NLI (GPU, on-demand)
  [Axis 3: Consistency] — Semantic entropy, N=3 samples
  [Axis 4: Equity ★] — DI + EOD, counterfactual substitution [BOLD, larger box to indicate importance]
  [Axis 5: Attribution] — Retrieval-attribution Jaccard@k
  Below all axes: [Drift Detector] — Page-Hinkley streaming CUSUM
  [AuditEnvelope] — aggregates all five axis results + drift signals

COLUMN 4 — OUTPUT (background: #EEE8F5, accent: #7C3AED):
Section header: "OUTPUT"
Boxes:
  [SQLite Logger] — per-response envelope persistence
  [Dashboard] — AuditEnvelopePanel live 5-axis radar + drift sparklines
  [arXiv Paper] — research contribution (dashed border to indicate publication artifact)

CONNECTIONS (every arrow individually):
1. Voice Query → Qwen3 8B: solid blue arrow, label "STT"
2. Text Prompt → Qwen3 8B: solid blue arrow
3. Retrieval Context → Qwen3 8B: solid blue arrow, label "RAG chunks"
4. Retrieval Context → Axis 5 Attribution: solid green arrow (retrieval also feeds attribution)
5. Qwen3 8B → Response: thick amber arrow (primary output flow)
6. Response → Axis 1 Calibration: solid green arrow
7. Response → Axis 2 Faithfulness: solid green arrow
8. Response → Axis 3 Consistency: solid green arrow (includes paraphrase generation)
9. Axis 3 Consistency → Axis 2 Faithfulness: thin dashed arrow labeled "reuses HHEM"
10. Response → Axis 4 Equity: solid green arrow, labeled "counterfactual sub."
11. Response + Retrieval → Axis 5 Attribution: solid green arrow
12. All Axes → AuditEnvelope: converging arrows
13. AuditEnvelope → Drift Detector: solid green arrow
14. Drift Detector → AuditEnvelope: thin feedback arrow labeled "alarm"
15. AuditEnvelope → SQLite Logger: violet arrow
16. AuditEnvelope → Dashboard: violet arrow
17. GPU Manager ↔ Axis 2 Faithfulness: thin dashed double-headed gray arrow labeled "load/unload"

CONSTRAINTS:
- No clip art, no stock icons, no emoji, no photorealistic elements
- No gradients — flat fills only
- Axis 4 Equity box should be slightly larger or have a subtle gold highlight border to indicate it is the novelty contribution
- "★" marker next to "Equity" label to mark it as the key novelty
- Paper is white, not cream or off-white
- All text must be EXACTLY as specified — do not abbreviate or rearrange labels
- The diagram should read cleanly at 6.75 inches wide (full NeurIPS column width)
- Professional academic figure aesthetic — NOT a marketing diagram
"""


def generate_image(prompt_text: str, attempt_num: int) -> str | None:
    print(f"\n{'='*60}\nAttempt {attempt_num}\n{'='*60}")
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt_text,
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        output_path = os.path.join(OUTPUT_DIR, f"fig_architecture_attempt{attempt_num}.png")
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                print(f"Saved: {output_path} ({os.path.getsize(output_path):,} bytes)")
                return output_path
            elif hasattr(part, "text") and part.text:
                print(f"Text response: {part.text[:300]}")
        print("WARNING: No image in response")
        return None
    except Exception as e:
        print(f"ERROR on attempt {attempt_num}: {e}")
        return None


def main() -> None:
    results = []
    for i in range(1, 4):
        if i > 1:
            time.sleep(3)
        path = generate_image(PROMPT, i)
        if path:
            results.append(path)
    if not results:
        print("\nAll attempts failed. Check GEMINI_API_KEY and model availability.")
        sys.exit(1)
    print(f"\nGenerated {len(results)} attempt(s). Review and rename best to fig_architecture.png")


if __name__ == "__main__":
    main()
