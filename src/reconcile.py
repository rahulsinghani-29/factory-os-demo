"""Evidence-keyed report routing. Reported output never enters confirmed receipt totals."""
import datetime as dt
import hashlib
import json
import os
import pathlib
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "data" / "wip_handoff_fixture.json"
LEDGER = ROOT / "logs" / "media-demo-ledger.jsonl"
LOCK = threading.Lock()
MODEL = os.environ.get("QUINCE_OLLAMA_MODEL", "qwen2.5:7b")
ENDPOINT = "http://localhost:11434/v1/chat/completions"

OUTPUT_FIELDS = ("po_id", "sku", "plan_version", "line", "shift", "granularity", "bundle_id",
                 "stage", "action", "quantity", "unit", "quantity_basis",
                 "interval_start", "interval_end")
MATCH_SCOPE_FIELDS = tuple(field for field in OUTPUT_FIELDS if field != "quantity")
REQUIRED_OUTPUT_FIELDS = tuple(field for field in OUTPUT_FIELDS if field != "bundle_id")
EVENT_TYPES = {"output_report", "qc_result_report", "exception_comment",
               "correction_proposal", "future_intent", "unclear"}


def read_fixture(path=FIXTURE):
    data = json.loads(pathlib.Path(path).read_text())
    if data.get("synthetic") is not True:
        raise ValueError("media demo records must be marked synthetic")
    return data


def read_ledger(path=LEDGER):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    entries = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"invalid event ledger JSON at line {line_no}") from exc
    return entries


def confirmed_receipt_total(fixture=None, ledger=None):
    fixture = fixture or read_fixture()
    ledger = ledger or []
    by_id = {}
    for event in fixture.get("confirmed_receipts", []):
        event_id = event.get("event_id")
        if not event_id:
            raise ValueError("confirmed receipt missing event_id")
        if event_id in by_id and by_id[event_id] != event:
            raise ValueError("conflicting confirmed receipt ID")
        by_id[event_id] = event
    # The local ledger is intentionally append-only and report-oriented. Even a
    # human-reviewed report, comment, or correction can never become a receipt here.
    return sum(int(event["quantity"]) for event in by_id.values()
               if event.get("event_type") == "confirmed_receipt" and event.get("status") == "confirmed")


def _present(candidate, keys):
    return all(candidate.get(key) not in (None, "") for key in keys)


def _exact_scope(candidate, existing):
    if candidate.get("granularity") == "bundle" and not candidate.get("bundle_id"):
        return False
    return all(candidate.get(k) == existing.get(k) for k in MATCH_SCOPE_FIELDS)


def _same_bundle_interval(candidate, existing):
    if candidate.get("bundle_id") != existing.get("bundle_id"):
        return False
    if candidate.get("po_id") != existing.get("po_id") or candidate.get("line") != existing.get("line"):
        return False
    try:
        left_start = dt.datetime.fromisoformat(candidate["interval_start"])
        left_end = dt.datetime.fromisoformat(candidate["interval_end"])
        right_start = dt.datetime.fromisoformat(existing["interval_start"])
        right_end = dt.datetime.fromisoformat(existing["interval_end"])
        return max(left_start, right_start) < min(left_end, right_end)
    except (KeyError, TypeError, ValueError):
        return False


def _same_interval(candidate, existing):
    return (candidate.get("interval_start") is not None and
            candidate.get("interval_start") == existing.get("interval_start") and
            candidate.get("interval_end") == existing.get("interval_end"))


def _existing_output_events(fixture, ledger):
    confirmed = list(fixture.get("confirmed_receipts", []))
    reported = []
    for entry in ledger:
        if entry.get("event_type") == "reported_output" and isinstance(entry.get("candidate"), dict):
            reported.append({**entry["candidate"], "event_id": entry.get("event_id"),
                             "source_hash": entry.get("source_hash"),
                             "stored_event_type": "reported_output"})
    return confirmed + reported


def reconcile_candidate(candidate, *, source_hash, fixture=None, ledger=None):
    """Return a route, matching evidence, and stable keys; never changes fixture totals."""
    fixture = fixture or read_fixture()
    ledger = ledger if ledger is not None else read_ledger()
    candidate = dict(candidate)
    event_type = candidate.get("event_type", "unclear")
    plan = fixture["plan"]
    if event_type not in EVENT_TYPES:
        event_type = "unclear"
        candidate["event_type"] = event_type
    candidate["source_hash"] = source_hash

    exact_replay = next((e for e in ledger if e.get("source_hash") == source_hash), None)
    if exact_replay:
        return {"route": "MATCH_ALREADY_RECORDED", "reason": "exact source hash already has a local reviewed ledger result",
                "match": exact_replay, "candidate": candidate, "needs_attention": False,
                "count_change": 0, "confirmed_total": confirmed_receipt_total(fixture, ledger)}

    if event_type == "future_intent" or candidate.get("quantity_basis") == "planned_target":
        return {"route": "NO_OP_FUTURE", "reason": "future intent or target is not an actual output event",
                "candidate": candidate, "needs_attention": False, "count_change": 0,
                "confirmed_total": confirmed_receipt_total(fixture, ledger)}

    if event_type == "unclear":
        return {"route": "UNCLEAR_OR_IRRELEVANT", "reason": "no supported event type was extracted",
                "candidate": candidate, "needs_attention": True, "count_change": 0,
                "clarifications": ["Is this a production event, a QC event, an exception comment, or irrelevant audio/image text?"],
                "confirmed_total": confirmed_receipt_total(fixture, ledger)}

    if event_type == "exception_comment":
        if not candidate.get("comment") or not (candidate.get("po_id") or candidate.get("line")):
            return _clarify(candidate, fixture, ledger,
                            ["Read either a PO number or a line, and preserve the reported comment."])
        scope = "purchase_order" if candidate.get("po_id") else "line"
        return {"route": "COMMENT_ONLY", "reason": "reported explanation; it does not change a count and its cause is not verified",
                "scope": scope,
                "candidate": candidate, "needs_attention": True, "count_change": 0,
                "clarifications": [], "confirmed_total": confirmed_receipt_total(fixture, ledger)}

    if event_type == "correction_proposal":
        event_ids = {e.get("event_id") for e in fixture.get("confirmed_receipts", [])}
        event_ids.update(e.get("event_id") for e in fixture.get("qc_events", []))
        event_ids.update(e.get("event_id") for e in ledger)
        if not candidate.get("linked_event_id") or candidate["linked_event_id"] not in event_ids:
            return _clarify(candidate, fixture, ledger, ["Link the proposed correction to an existing event ID or leave unresolved."])
        return {"route": "CORRECTION_REVIEW", "reason": "correction is a separate linked proposal; original event remains unchanged",
                "candidate": candidate, "match": {"event_id": candidate["linked_event_id"]},
                "needs_attention": True, "count_change": 0, "clarifications": [],
                "confirmed_total": confirmed_receipt_total(fixture, ledger)}

    if event_type == "qc_result_report":
        link = candidate.get("linked_event_id")
        related = [e for e in fixture.get("qc_events", []) if
                   (link and e.get("linked_receipt_id") == link) or
                   (candidate.get("bundle_id") and e.get("bundle_id") == candidate.get("bundle_id") and
                    candidate.get("action") == e.get("action") and candidate.get("quantity") == e.get("quantity"))]
        if related:
            return {"route": "MATCH_ALREADY_RECORDED", "reason": "matching linked QC event already exists; no QC or count write",
                    "match": related[0], "candidate": candidate, "needs_attention": False,
                    "count_change": 0, "confirmed_total": confirmed_receipt_total(fixture, ledger)}
        missing = [key for key in ("po_id", "bundle_id", "stage", "action", "quantity", "unit") if not _present(candidate, (key,))]
        if missing:
            return _clarify(candidate, fixture, ledger, ["Clarify QC bundle, stage, disposition, quantity, and unit."])
        return {"route": "MISSING_RECORD", "reason": "unmatched QC statement remains a reported QC proposal, not a disposition",
                "candidate": candidate, "needs_attention": True, "count_change": 0,
                "clarifications": ["Confirm the QC actor and event time; no disposition will be overwritten."],
                "confirmed_total": confirmed_receipt_total(fixture, ledger)}

    # Output and receipt-shaped claims are always treated as reports unless they
    # match a pre-existing confirmed handoff with the full compatible event scope.
    missing = [key for key in REQUIRED_OUTPUT_FIELDS if not _present(candidate, (key,))]
    if candidate.get("granularity") == "bundle" and not candidate.get("bundle_id"):
        missing.append("bundle_id")
    if candidate.get("quantity_basis") != "incremental":
        missing.append("quantity_basis (incremental vs cumulative/target is unresolved)")
    if candidate.get("unit") != "garments":
        missing.append("unit conversion to garments")
    if candidate.get("quantity") is not None and (isinstance(candidate["quantity"], bool) or not isinstance(candidate["quantity"], (int, float)) or candidate["quantity"] < 0):
        missing.append("valid nonnegative quantity")
    if candidate.get("layout_uncertain") or candidate.get("ocr_uncertain") or candidate.get("header_missing") or candidate.get("crossed_out"):
        missing.append("human verification of uncertain photo cell/layout")
    if candidate.get("uncertainty"):
        missing.append("human verification of uncertain speech/number")
    if missing:
        return _clarify(candidate, fixture, ledger,
                        ["Clarify before proposing a reported-output record: " + ", ".join(dict.fromkeys(missing)) + "."])

    scope_matches = [e for e in _existing_output_events(fixture, ledger) if _exact_scope(candidate, e)]
    if scope_matches:
        same_qty = next((e for e in scope_matches if e.get("quantity") == candidate.get("quantity")), None)
        if same_qty:
            return {"route": "MATCH_ALREADY_RECORDED", "reason": "matching full event scope and quantity; separate source is a no-op",
                    "match": same_qty, "candidate": candidate, "needs_attention": False,
                    "count_change": 0, "confirmed_total": confirmed_receipt_total(fixture, ledger)}
        return _conflict(candidate, fixture, ledger, scope_matches[0], "same full event scope has a different reported quantity")

    possible = [e for e in _existing_output_events(fixture, ledger)
                if _same_bundle_interval(candidate, e) and _same_interval(candidate, e)]
    if possible:
        return _conflict(candidate, fixture, ledger, possible[0],
                         "same bundle and overlapping interval, but stage/action/granularity differs")

    granularity_conflicts = [e for e in _existing_output_events(fixture, ledger)
                             if candidate.get("po_id") == e.get("po_id") and
                             candidate.get("line") == e.get("line") and
                             candidate.get("granularity") != e.get("granularity") and
                             "line_hour" in (candidate.get("granularity"), e.get("granularity")) and
                             _same_bundle_interval({**candidate, "bundle_id": e.get("bundle_id")}, e)]
    if granularity_conflicts:
        return _conflict(candidate, fixture, ledger, granularity_conflicts[0],
                         "line-hour and bundle events overlap; their different granularity cannot be summed or deduplicated")

    semantic_key = _semantic_key(candidate)
    report_match = next((e for e in _existing_output_events(fixture, ledger)
                         if e.get("stored_event_type") == "reported_output" and
                         e.get("semantic_key") == semantic_key and e.get("quantity") == candidate.get("quantity")), None)
    if report_match:
        return {"route": "MATCH_ALREADY_RECORDED", "reason": "same reported event already exists from another source; linked no-op",
                "match": report_match, "candidate": candidate, "needs_attention": False,
                "count_change": 0, "confirmed_total": confirmed_receipt_total(fixture, ledger)}
    return {"route": "MISSING_RECORD", "reason": "no matching confirmed handoff; draft only as reported output after review",
            "candidate": candidate, "needs_attention": True, "count_change": 0,
            "clarifications": ["Confirm actor identity, event time/interval, unit, and action. Approval can save only a reported record, never a verified receipt."],
            "confirmed_total": confirmed_receipt_total(fixture, ledger)}


def _clarify(candidate, fixture, ledger, questions):
    return {"route": "NEEDS_CLARIFICATION", "reason": "required identity, event scope, unit, or evidence is missing/uncertain",
            "candidate": candidate, "needs_attention": True, "count_change": 0,
            "clarifications": questions, "confirmed_total": confirmed_receipt_total(fixture, ledger)}


def _conflict(candidate, fixture, ledger, existing, reason):
    return {"route": "CONFLICT", "reason": reason, "candidate": candidate,
            "existing_event": existing, "needs_attention": True, "count_change": 0,
            "clarifications": ["Choose a distinct scope/time, request a physical check, propose a linked correction, or leave unresolved. Both source events remain preserved."],
            "confirmed_total": confirmed_receipt_total(fixture, ledger)}


def _semantic_key(candidate):
    fields = OUTPUT_FIELDS + ("event_type",)
    packed = json.dumps({key: candidate.get(key) for key in fields}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def validate_review_decision(reconciliation, decision, edits=None):
    """Validate reviewer payload after LangGraph interrupt; no side effects here."""
    edits = edits or {}
    candidate = dict(reconciliation["candidate"])
    candidate.update(edits)
    route = reconciliation["route"]
    if decision == "reject":
        return {"valid": True, "write": None, "review_decision": "rejected", "candidate": candidate}
    if route == "COMMENT_ONLY" and decision == "confirm_comment":
        if not str(candidate.get("comment", "")).strip():
            return {"valid": False, "error": "comment text cannot be blank"}
        return {"valid": True, "write": "exception_comment", "review_decision": "confirmed_comment", "candidate": candidate}
    if route == "MISSING_RECORD" and decision == "save_reported_output":
        required = ["po_id", "sku", "plan_version", "line", "shift", "granularity", "stage", "action",
                    "quantity", "unit", "quantity_basis", "interval_start", "interval_end", "actor_id"]
        missing = [key for key in required if candidate.get(key) in (None, "")]
        if missing:
            return {"valid": False, "error": "cannot save reported output; clarify fields: " + ", ".join(missing)}
        if candidate.get("unit") != "garments" or candidate.get("quantity_basis") != "incremental":
            return {"valid": False, "error": "unit conversion or cumulative/incremental status remains unresolved"}
        if candidate.get("granularity") == "bundle" and not candidate.get("bundle_id"):
            return {"valid": False, "error": "bundle ID required at bundle granularity"}
        return {"valid": True, "write": "reported_output", "review_decision": "saved_as_reported",
                "candidate": candidate}
    if route == "CORRECTION_REVIEW" and decision == "save_correction_proposal":
        if not candidate.get("linked_event_id") or not candidate.get("actor_id") or not candidate.get("event_time"):
            return {"valid": False, "error": "correction needs linked event, responsible actor, and event time"}
        return {"valid": True, "write": "correction_proposal", "review_decision": "saved_correction_proposal",
                "candidate": candidate}
    if route == "CONFLICT" and decision in ("leave_unresolved", "request_physical_check", "propose_linked_correction"):
        if decision == "propose_linked_correction" and not edits.get("linked_event_id"):
            return {"valid": False, "error": "select an existing linked event before proposing a correction"}
        if decision == "propose_linked_correction":
            candidate["linked_event_id"] = edits["linked_event_id"]
            existing = reconciliation.get("existing_event") or reconciliation.get("match") or {}
            if candidate["linked_event_id"] != existing.get("event_id"):
                return {"valid": False, "error": "correction must link to the conflicting existing event"}
            if not edits.get("actor_id") or not edits.get("event_time"):
                return {"valid": False, "error": "linked correction proposal needs a responsible actor and event time"}
            candidate["actor_id"] = edits["actor_id"]
            candidate["event_time"] = edits["event_time"]
        if decision == "request_physical_check" and not candidate.get("actor_id"):
            return {"valid": False, "error": "record the responsible actor for the requested physical check"}
        write = (None if decision == "leave_unresolved" else
                 "correction_proposal" if decision == "propose_linked_correction" else decision)
        return {"valid": True, "write": write,
                "review_decision": decision, "candidate": candidate}
    if route == "NEEDS_CLARIFICATION" and decision in ("save_clarification", "reject"):
        if decision == "save_clarification" and not str(edits.get("clarification", "")).strip():
            return {"valid": False, "error": "enter a specific clarification question"}
        if decision == "save_clarification":
            candidate["clarification"] = str(edits["clarification"]).strip()
        return {"valid": True, "write": "clarification" if decision == "save_clarification" else None,
                "review_decision": decision, "candidate": candidate}
    if route == "UNCLEAR_OR_IRRELEVANT" and decision in ("reject", "ask_clarification"):
        if decision == "ask_clarification" and not str(edits.get("clarification", "")).strip():
            return {"valid": False, "error": "enter a specific clarification question"}
        if decision == "ask_clarification":
            candidate["clarification"] = str(edits["clarification"]).strip()
        return {"valid": True, "write": "clarification" if decision == "ask_clarification" else None,
                "review_decision": decision, "candidate": candidate}
    return {"valid": False, "error": "review decision is not permitted for this route"}


def append_reviewed_event(run_id, source_hash, proposal_index, decision, validation,
                          ledger_path=LEDGER, recorded_at=None):
    """Append only a reviewed non-receipt record; repeated commits are idempotent."""
    write_kind = validation.get("write")
    if not validation.get("valid") or not write_kind:
        return {"written": False, "reason": "no approved local record for this decision"}
    candidate = validation["candidate"]
    key = f"{source_hash}:{proposal_index}:{write_kind}"
    ledger_path = pathlib.Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        entries = read_ledger(ledger_path)
        prior = next((event for event in entries if event.get("idempotency_key") == key), None)
        if prior:
            return {"written": False, "reason": "duplicate commit suppressed", "event_id": prior["event_id"]}
        event_id = "DEMO-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16].upper()
        if write_kind == "exception_comment":
            event_type, status = "exception_comment", "reported"
        elif write_kind == "reported_output":
            event_type, status = "reported_output", "reported_unverified"
        elif write_kind == "correction_proposal":
            event_type, status = "correction_proposal", "proposed"
        elif write_kind == "clarification":
            event_type, status = "clarification", "open"
        else:
            event_type, status = "review_followup", "reported"
        output = {"event_id": event_id, "idempotency_key": key, "event_type": event_type,
                  "status": status, "source_hash": source_hash, "source_proposal_index": proposal_index,
                  "run_id": run_id, "recorded_at": recorded_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "review_decision": validation.get("review_decision"), "candidate": candidate,
                  "linked_event_id": candidate.get("linked_event_id"), "confirmed_receipt": False,
                  "human_operational_approval": False, "outbox_written": False}
        with ledger_path.open("a") as handle:
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {"written": True, "event_id": event_id, "event_type": event_type,
                "status": status, "confirmed_receipt": False}
