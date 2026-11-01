# Architecture and product choices

The design accepts a continuing stream of factory updates. The implemented demo runs one supplied photo or text event at a time. A live WhatsApp webhook and durable stream processor are not connected.

```mermaid
flowchart LR
  A[Photo or text message] --> B[Read input locally]
  B --> C[Load editable PO, plan, handoff and prior-message context]
  C --> D[Calculate pace in Python]
  D --> E[Qwen: interpret the note against the facts]
  E --> F[Check quotes, IDs and uncertainty]
  F --> G[One short report]
  G --> H[Clear scoped sewing risk]
  G --> I[Check input or supervisor follow-up]
```

The actual compiled graph is built in `src/live_workflow.py`. The Workflow design page lets a reviewer inspect and download its Mermaid representation and view the exact prompts.

## Model responsibilities

Gemma 3 4B transcribes the photo, including Tamil remarks. Text inputs preserve the original message. Qwen 2.5 7B receives that text and the supplied source records. Expected labels are excluded.

The model interprets language. It does not decide arithmetic truth. Python parses the incremental count column, checks row quotes, reconciles the total, and calculates rates within explicit working windows. PO and line must match the context. A forecast is scoped to reported sewing output; it does not create a confirmed handoff or shipment fact.

The report builder provides one user-facing verdict. A date conflict becomes “Check the date,” with a clearly stated pace estimate under the supplied date assumption. Detailed math and raw model responses stay available for inspection.

## Operational boundary

The application has no factory-write or send tool. A supervisor assignment is a suggested follow-up, not an integrated task. Historical CLI records are not included in this public copy. No web result is presented as human approval.

## Next steps

Connect authenticated message intake and real PO/plan/handoff APIs; version each source snapshot; replace OCR and prompts only after a reviewed media benchmark; add durable background jobs and real task creation; measure language accuracy and latency by input type. These are roadmap items, not shipped functionality.
