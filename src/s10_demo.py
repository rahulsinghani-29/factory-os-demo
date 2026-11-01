"""Synthetic S10 hourly-sheet example and compiled multi-source LangGraph."""
import json
import pathlib
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from llm import call_ollama_json
from reconcile import MODEL

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = """You are a factory status assistant. Use only the supplied synthetic source records.
The inputs are: (1) plan API target for this shift, (2) PO details API with total requirement and due date,
(3) WhatsApp text updates, (4) the hourly S10 sheet sent as a WhatsApp photo, and (5) department handoffs
for cutting, finishing, and QC. WhatsApp voice is a supported future channel, but this demo has no voice
recording or transcript. The hourly sheet is the live production report. Its total is the sum of the
hourly rows; do not add a printed total twice.
Use the supplied as-of timestamp. Separate current status from uncertain future risk.
Separate reported output from handoff-confirmed output. A planned date is not an actual event. Forecasts
are uncertain. Do not invent effects on sewing or shipment, owners, or actions. Do not claim delay from a
same-day target or missing confirmation. If sources conflict, QC is blurry, or required due-date/plan facts
are missing, return outcome=unclear, delay_risk=null, assign_to=line_supervisor. Otherwise return
outcome=clear and delay_risk=yes or no, with a short explanation limited to the supplied facts.
Return JSON only: {"outcome":"clear|unclear","delay_risk":"yes|no|null","explanation":"...",
"evidence_ids":[...],"assign_to":"line_supervisor|null"}. Every evidence ID must come from input."""


def load_examples():
    return json.loads((ROOT / "data" / "s10_shift_examples.json").read_text())


def validate_source_bundle(example):
    """Fail to the supervisor route when evidence is insufficient or inconsistent."""
    reasons = []
    rows = example.get("rows", [])
    row_total = sum(row.get("sewn", 0) for row in rows if isinstance(row.get("sewn"), int))
    if len(rows) != 6 or row_total != example.get("total_sewn_so_far"):
        reasons.append("The six visible hourly entries do not reconcile to the stated running total.")
    po = example.get("po_details_api") or {}
    plan = example.get("plan_api") or {}
    if not po.get("due_at") or not plan.get("target_units_by_16:00"):
        reasons.append("Due date or shift plan is missing.")
    if any((item.get("blurry") for item in (example.get("handoffs") or {}).values())):
        reasons.append("A department handoff is marked blurry.")
    updates = example.get("whatsapp_updates", [])
    if len(updates) > 1:
        reasons.append("Updates conflict or need reconciliation.")
    return {"clear": not reasons, "reasons": reasons, "hourly_sum": row_total}


class ShiftState(TypedDict, total=False):
    example: dict
    checks: dict
    hourly_sheet: dict
    plan_api: dict
    po_details_api: dict
    updates_handoffs: dict
    whatsapp_text: dict
    whatsapp_photo: dict
    whatsapp_voice: dict
    handoffs: dict
    model_result: dict
    outcome: dict
    log_path: str


def _read_whatsapp_photo(state):
    example = state["example"]
    return {"whatsapp_photo": {"channel": "WhatsApp photo", "line": "S10", "rows": example["rows"],
                              "total_sewn_so_far": example["total_sewn_so_far"],
                              "source_id": example.get("whatsapp_photo_source_id"),
                              "ocr_status": "fixture text; image OCR not run"}}


def _receive_whatsapp_stream(state):
    """Represent the continuous upstream channel; this demo processes one selected sample."""
    return {"whatsapp_stream": {"mode": "continuous upstream channel",
                                "demo_scope": "one synthetic example per run"}}


def _read_whatsapp_text(state):
    return {"whatsapp_text": {"channel": "WhatsApp text", "messages": state["example"]["whatsapp_updates"]}}


def _read_whatsapp_voice(state):
    return {"whatsapp_voice": {"channel": "WhatsApp voice", "status": "not_demonstrated",
                               "transcript": None}}


def _read_plan(state):
    return {"plan_api": state["example"]["plan_api"]}


def _read_po(state):
    return {"po_details_api": state["example"]["po_details_api"]}


def _read_updates_handoffs(state):
    example = state["example"]
    return {"handoffs": example["handoffs"]}


def _infer(state):
    example = state["example"]
    checks = state["checks"]
    if not checks["clear"]:
        return {"outcome": {"outcome": "unclear", "delay_risk": None,
            "explanation": " ".join(checks["reasons"]), "evidence_ids": example["evidence_ids"],
            "assign_to": "line_supervisor", "source": "deterministic evidence gate"}}
    log_path = pathlib.Path(state.get("log_path") or ROOT / "logs" / "media-demo" / "s10-inference.json")
    inference_inputs = {
        "work_date": example.get("work_date"), "as_of": example.get("as_of"),
        "whatsapp_photo_hourly_sheet": {"line": "S10", "rows": example["rows"],
            "total_sewn_so_far": example["total_sewn_so_far"],
            "source_id": example.get("whatsapp_photo_source_id"),
            "ocr_status": "fixture text; image OCR not run"},
        "whatsapp_text_updates": example["whatsapp_updates"],
        "whatsapp_voice": {"status": "not_demonstrated", "transcript": None},
        "plan_api": example["plan_api"], "po_details_api": example["po_details_api"],
        "department_handoffs": example["handoffs"], "evidence_ids": example["evidence_ids"]}
    try:
        response = call_ollama_json(PROMPT, json.dumps(inference_inputs, ensure_ascii=False), MODEL,
                                    log_path, timeout=90, max_tokens=220)
        parsed = json.loads(response["text"])
        allowed_ids = set(example["evidence_ids"])
        valid = (isinstance(parsed, dict) and parsed.get("outcome") == "clear" and
                 parsed.get("delay_risk") in ("yes", "no") and
                 parsed.get("assign_to") is None and isinstance(parsed.get("explanation"), str) and
                 isinstance(parsed.get("evidence_ids"), list) and
                 set(parsed["evidence_ids"]).issubset(allowed_ids) and parsed["evidence_ids"])
        if not valid:
            raise ValueError("Model output failed the evidence-linked outcome schema")
        result = {**parsed, "model_requested": MODEL, "model_returned": response.get("model", MODEL),
                  "log_file": str(response.get("log_file")), "endpoint": response.get("endpoint")}
    except Exception as exc:
        result = {"outcome": "unclear", "delay_risk": None,
                  "explanation": "Local inference did not produce a valid, evidence-linked result. " +
                                 type(exc).__name__, "evidence_ids": example["evidence_ids"],
                  "assign_to": "line_supervisor", "model_requested": MODEL,
                  "model_returned": None, "log_file": None, "endpoint": "http://localhost:11434/v1/chat/completions"}
    return {"outcome": result}


def _route(state):
    result = state.get("outcome") or {}
    return "clear" if result.get("outcome") == "clear" else "unclear"


def _end_clear(state):
    return {"outcome": {**state["outcome"], "terminal": "CLEAR"}}


def _end_unclear(state):
    return {"outcome": {**state["outcome"], "terminal": "UNCLEAR · HITL"}}


def build_shift_graph():
    graph = StateGraph(ShiftState)
    graph.add_node("receive_whatsapp_stream", _receive_whatsapp_stream)
    graph.add_node("read_whatsapp_photo", _read_whatsapp_photo)
    graph.add_node("read_whatsapp_text", _read_whatsapp_text)
    graph.add_node("read_whatsapp_voice", _read_whatsapp_voice)
    graph.add_node("read_plan_api", _read_plan)
    graph.add_node("read_po_details_api", _read_po)
    graph.add_node("read_department_handoffs", _read_updates_handoffs)
    graph.add_node("compare_evidence", lambda s: {"checks": validate_source_bundle(s["example"])})
    graph.add_node("infer_from_grounded_sources", _infer)
    graph.add_node("end_clear", _end_clear)
    graph.add_node("end_unclear_supervisor", _end_unclear)
    graph.add_edge(START, "receive_whatsapp_stream")
    graph.add_edge("receive_whatsapp_stream", "read_whatsapp_photo")
    graph.add_edge("receive_whatsapp_stream", "read_whatsapp_text")
    graph.add_edge("receive_whatsapp_stream", "read_whatsapp_voice")
    graph.add_edge(START, "read_plan_api")
    graph.add_edge(START, "read_po_details_api")
    graph.add_edge(START, "read_department_handoffs")
    graph.add_edge(["read_whatsapp_photo", "read_whatsapp_text", "read_whatsapp_voice",
                    "read_plan_api", "read_po_details_api", "read_department_handoffs"], "compare_evidence")
    graph.add_edge("compare_evidence", "infer_from_grounded_sources")
    graph.add_conditional_edges("infer_from_grounded_sources", _route,
        {"clear": "end_clear", "unclear": "end_unclear_supervisor"})
    graph.add_edge("end_clear", END)
    graph.add_edge("end_unclear_supervisor", END)
    return graph.compile()
