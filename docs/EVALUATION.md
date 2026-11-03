# Evaluation rubric

A reviewer scores a run against the original image transcription, source records, and correct risk decision. Each criterion scores 0, 1 or 2. Total score is the sum of `weight × score / 2`.

| Criterion | Weight | 0 · Fail | 1 · Partial | 2 · Pass |
|---|---:|---|---|---|
| Identifier and time accuracy | 15 | Wrong PO, line or time | One missing identifier, explicitly unknown | Correct PO, line, date, hour and timezone |
| Counts, rates and finish math | 20 | Wrong count, double count, blank treated as zero | Counts correct but calendar/rate assumptions incomplete | Exact sum and rates, blanks excluded, work hours and finish calculation correct |
| Tamil remarks and event meaning | 20 | Missed stoppage or intent treated as completion | Core issue captured, nuance missing | Original remark preserved and meaning correct |
| Cross-source comparison | 15 | Mismatched POs or ignored conflicts | Some sources or differences missed | Links the right records and explains conflicts/unknowns |
| Risk and abstention | 20 | False clear or invented delay | Safe unclear on a decidable case | Correct scoped risk decision or necessary abstention |
| Evidence and explanation | 10 | Fabricated quote or unsupported cause | Real sources, incomplete support or verbose output | Exact quotes and IDs, short explanation and stated limits |

## Proposed release gate

Require 90/100 and no critical error. A wrong PO/line, invented source, invented shipment delay, future plan treated as completed, false clear on conflicting facts, or unauthorized write fails the run regardless of total score.

Version the source image, exact transcription, quantities, remark translation, source snapshot, expected decision and rationale. Split by template and PO to avoid near-duplicate leakage. Cover clean print, handwriting, Tamil, mixed language, zero-output hours, blank future hours, partial work, stopped machines, conflicting updates, missing dates, and unreadable photos.

Report identifier exact match, row-count accuracy, character error rate, remark meaning accuracy, risk precision/recall, false-clear rate, abstention rate and unnecessary abstention. Measure OCR, LLM and total p50/p95 latency separately. Proposed targets are 98% identifier/count accuracy, 95% risk recall, at most 1% false clear, and zero fabricated evidence or writes. They are not measured achievements.

The 15-second warm OCR target has not been met consistently. Local API charge is zero; hardware cost is unmeasured. A reviewed media golden set and automatic sampled evaluator are not implemented. The proposed async policy is to score 10% of live runs plus every unclear/failed run, and to run the fixed golden set after every relevant change.

The 23 shared offline regression tests are useful checks, not a production OCR benchmark.

See [the ten authored reference cases](../data/golden_cases.json) and [the workflow/reference notes](workflow-and-reference-cases.md). All references are pending independent review. G08 and G10 describe known gaps. No model score is implied by listing expected labels.
