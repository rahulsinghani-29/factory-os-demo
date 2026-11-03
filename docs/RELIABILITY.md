# Evidence and reliability limits

This public copy contains application code, generated synthetic S10 sheets, invented source context and code-based regression tests. It excludes raw run logs, original floor notes, historical source files, private uploads and local environments. No credentials or private keys are included.

## Observed during local development

- Real local photo OCR and Qwen calls completed on generated sheets. Counts were usable in tested examples, but dates, Tamil and future blank cells were sometimes misread.
- One on-track image produced 400 reported pieces. The lowest observed rate was 62 pieces/hour, with 18 planned work hours available. The numerical forecast met the 1,000-piece requirement, but OCR misread the document year. The report therefore required the date to be checked.
- A stopped-machine example contained 188 pieces over six observed hours. Its average-rate projection was 752 pieces by the due date, indicating risk against 1,000 required. Future rows were excluded even when OCR inserted zeros into blank cells.
- Real Qwen text runs completed. Unknown restart time no longer hides a valid conditional rate calculation. This does not confirm a restart, QC clearance or shipment ETA.
- The full local workspace passed 89 offline tests. This public subset has 23 standalone tests; historical tests depending on withheld source logs are excluded.

These are observed development results, not a production accuracy benchmark. Raw logs remain local rather than being published as potentially sensitive data. Public tests use invented values and mocked clients to reproduce the relevant classes of failure.

## Operational boundaries

The web app has no factory-write tool. Reported sewing output is not a confirmed handoff receipt. A supervisor follow-up does not open a real task. Automated tests are not human approval.

Exact-quote, identifier, total and calendar checks do not prove that a Tamil translation is correct. The model can misread inputs, invent an explanation or abstain unnecessarily. The report uses the checked calculation and keeps uncertainty visible.

Live WhatsApp ingestion, voice in the web path, factory APIs, a durable worker, a reviewed media gold set, automatic sampled evaluation and a real supervisor queue are not connected. Hosted Langfuse receipt is not independently verified. Some OCR runs exceeded the 15-second target.
