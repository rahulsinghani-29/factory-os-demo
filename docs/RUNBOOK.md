# Run and inspect the demo

Use Python 3.11+, Ollama, `gemma3:4b`, and `qwen2.5:7b`. Follow the commands in the root README. The web path uses these exact local model names; the current UI does not have a model picker. No keys are needed for local inference.

## Input

Choose a sample sheet, upload JPG/PNG, or paste text. Edit the source-context JSON to match its PO, line and event date. The forecast needs:

- An explicit `as_of` time and timezone.
- `po_details_api.po_id`, `total_required`, `due_at` and a source ID.
- `plan_api.line`, `sewn_before_shift` and dated `working_windows` with source ID.
- Optional shift end/target, prior WhatsApp messages, and handoff records.

Example pasted text for the default steady-work context:

```text
PO-SYN-101 line S10, as of 2026-11-01 14:00 IST.
08:00–09:00: 72
09:00–10:00: 70
10:00–11:00: 68
11:00–12:00: 65
12:00–13:00: 62
13:00–14:00: 63
Work continues. No stoppage is reported.
```

Custom photos actually run OCR. They do not use the fixture transcription. Future blank hours are excluded; any values that OCR puts in future hours are ignored and reported. A missing due date or calendar is not silently replaced with a default.

Run opens Output and shows a timer. The completed trace comes first, followed by one short conclusion. Open “See the calculation” for pace scenarios, “Check the source text” for OCR, and “Run details and model notes” for raw evidence.

## Local records

Each run writes `logs/live-demo/<run-id>/input.json`, `ocr.json` for a photo, `inference.json`, and `run-outcome.json`. These files can contain the user's input and are ignored by Git. Do not commit private input or generated logs.

Failed stages are recorded and cannot write factory data. Keep the model response and its validation result distinct from human approval.

## Optional Langfuse

Local tracing is always enabled. For cloud export, set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` in the shell that starts the app. Use the base URL for your Langfuse project region. Do not paste secrets into source, GitHub, screenshots or chat.

This command prompts without echoing values or storing them in shell history:

```bash
python3 -c 'import getpass,os; os.environ["LANGFUSE_PUBLIC_KEY"]=getpass.getpass("Public key: "); os.environ["LANGFUSE_SECRET_KEY"]=getpass.getpass("Secret key: "); os.environ["LANGFUSE_BASE_URL"]=getpass.getpass("Base URL: "); os.execv("./run_media_demo.sh",["./run_media_demo.sh"])'
```

The app displays configuration status. Export is opt-in and restricted to inputs marked synthetic. Settings present do not prove delivery. This repository contains no project keys.
