# AI tools and engineering choices

## Model use

- **Gemma 3 4B through local Ollama:** reads actual images, including counts and Tamil remarks. It can misread dates, handwriting and blanks.
- **Qwen 2.5 7B through local Ollama:** interprets the message against the supplied PO, calendar, handoffs and prior notes. It can over-abstain or produce unsupported explanations.
- **AI coding assistance:** helped develop the UI, prompts, graph, tests and documents. Generated code was checked with offline regression tests and selected real local-model runs.
- **Image generation:** produced the camera-style synthetic sheets. These are not photographs from a real factory and their text is not an independently reviewed language benchmark.

## What changed after failures

The earlier model output promoted future/partial work to completion and inferred unsupported delays. Those outputs remain preserved locally. This public copy uses invented test data and excludes raw historical logs. The current path keeps source quotes, separates plans from actual events, and uses explicit date and identifier checks.

The model initially abstained whenever restart time was unknown, even when a useful rate calculation was possible. Forecast math now runs in ordinary Python. It uses actual transcribed hourly rows and planned work windows, excluding future rows. It does not receive the examples' expected labels.

Separate arithmetic and LLM verdicts also confused the report. One report builder now owns the displayed conclusion. The model adds source interpretation; it cannot invent forecast numbers or contradict the accepted calculation on the main screen. Date conflicts remain visible rather than being hidden to force a clear answer.

## Human and operational limits

All sample data is synthetic. No financial benefit, production reliability, translation accuracy or business impact is established. A model response, accepted schema, automated test or suggested supervisor follow-up is not human approval. The web app has no factory-write capability.
