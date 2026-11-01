"""Source-linked extraction, deterministic alert drafting, and a fail-closed review gate."""
import csv
import datetime as dt
import json
import pathlib
import re
from llm import call_llm

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVENT_TYPES = ["cutting_start", "cutting_progress", "cutting_done", "sewing_start",
               "packing_done", "qc_result", "material_status", "approval", "other"]
EVENT_STATUSES = ["future_intent", "in_progress", "completed", "incomplete", "unknown"]
MILESTONES = ["fabric_ready", "trims_ready", "approval", "cutting_start", "cutting", "sewing_start",
              "packing", "ship", "qc", "other"]
PLAN_FIELDS = {m: m + "_plan" for m in MILESTONES if m + "_plan" in {
    "fabric_ready_plan", "trims_ready_plan", "approval_plan", "cutting_start_plan",
    "sewing_start_plan", "packing_plan", "ship_plan"}}
PLAN_FIELDS.update({"fabric_ready": "fabric_ready_plan", "trims_ready": "trims_ready_plan",
                    "approval": "approval_plan", "cutting_start": "cutting_start_plan",
                    "cutting": "cutting_start_plan",
                    "sewing_start": "sewing_start_plan", "packing": "packing_plan",
                    "ship": "ship_date_plan"})


def parse_json_loose(text):
    t = text.strip()
    candidates = [t, re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        return json.loads(t[i:j + 1])
    raise ValueError("no JSON object found in model output")


def load_orders():
    with (ROOT / "data" / "orders.csv").open(newline="") as f:
        return {row["po_id"]: row for row in csv.DictReader(f)}


def parse_source_records(text):
    """Read source headers and keep their body text, IDs, language and report time intact."""
    header = re.compile(r"^\[source_id=([^;]+); observed_at=([^;]+); language=([^;]+); source=(.+)\]$")
    records, current, body = [], None, []
    def flush():
        if current:
            current["text"] = "\n".join(body).strip()
            current["po_ids"] = sorted(set(re.findall(r"[QN]-\d{4}", current["text"])))
            records.append(current)
    for line in text.splitlines():
        match = header.match(line)
        if match:
            flush()
            current = {"source_id": match.group(1), "observed_at": match.group(2),
                       "language": match.group(3), "source": match.group(4)}
            body = []
        elif current:
            body.append(line)
    flush()
    return records


S1_SYSTEM = """Extract production facts from the supplied, source-ID-labelled notes.
Return strict JSON: {"events":[...]}. Each event must contain source_id, po_id,
event_type, milestone, event_status, quantity (number or null), quantity_basis
(reported_actual, planned_target, order_total, or unknown), occurred_at (only an explicit
event time/date, otherwise null), source_quote (exact contiguous source text), confidence,
needs_human_review, uncertainty_notes. event_status is one of: future_intent, in_progress,
completed, incomplete, unknown. event_type is one of: """ + ", ".join(EVENT_TYPES) + """.
milestone is one of: """ + ", ".join(MILESTONES) + """.
Rules: The source report time is not the event occurrence time. A promise or future-tense
statement is not progress or completion. A quantity mentioned as an order total is not a
completed quantity. Preserve partial quantities as partial; do not promote them to complete.
Keep the exact quote and matching source_id. If a source is ambiguous, blurry, mixed-language,
conflicting, or does not support status/quantity, use unknown/future_intent as appropriate
and set needs_human_review=true. Confidence never overrides evidence."""


def stage1_structure(records, valid_pos, as_of):
    user = json.dumps({"as_of": as_of, "valid_po_ids": valid_pos, "sources": records}, ensure_ascii=False)
    out = call_llm("stage1-structure", S1_SYSTEM, user)
    return {"parsed": parse_json_loose(out["text"]), "meta": out["meta"]}


def supplement_explicit_source_events(records):
    """Parse only narrow, literal clerk status rows that the model can omit or misclassify."""
    events = []
    for source in records:
        if source.get("language") != "en":
            continue
        for line in source["text"].splitlines():
            po_match = re.search(r"([QN]-\d{4})", line)
            if not po_match:
                continue
            po = po_match.group(1)
            common = {"source_id": source["source_id"], "po_id": po, "occurred_at": None,
                      "source_quote": line.strip(), "confidence": 1.0,
                      "needs_human_review": False, "uncertainty_notes": None}
            lower = line.lower()
            if "trims not received" in lower:
                events.append({**common, "event_type": "material_status", "milestone": "trims_ready",
                    "event_status": "incomplete", "quantity": None, "quantity_basis": "unknown"})
            elif "fabric received" in lower:
                events.append({**common, "event_type": "material_status", "milestone": "fabric_ready",
                    "event_status": "completed", "quantity": None, "quantity_basis": "unknown"})
            else:
                cut_match = re.search(r"cut pieces today:\s*([\d,]+)", line, re.I)
                if cut_match:
                    events.append({**common, "event_type": "cutting_progress", "milestone": "cutting",
                        "event_status": "in_progress", "quantity": int(cut_match.group(1).replace(",", "")),
                        "quantity_basis": "reported_actual"})
    return events


def merge_source_events(model_events, explicit_events):
    """Keep a literal source parse when present; avoid duplicate model versions of that fact."""
    explicit_keys = {(e["source_id"], e["po_id"], e["milestone"]) for e in explicit_events}
    merged = [e for e in model_events if (e.get("source_id"), e.get("po_id"), e.get("milestone")) not in explicit_keys]
    return merged + explicit_events


def _norm(text):
    return " ".join(str(text).split()).strip(" \"'“”‘’")


def _date(value):
    return dt.date.fromisoformat(str(value)[:10])


def _future_or_split_quote(quote, event, order):
    """Detect source-level contradictions using status, role, and the quoted quantity context."""
    q = event.get("quantity")
    if event.get("event_status") == "completed" and event.get("quantity") is not None and event.get("quantity_basis") != "reported_actual":
        return "completed quantity is not identified as a reported actual"
    if event.get("event_status") == "completed" and event.get("milestone") in ("cutting_start", "cutting"):
        numbers = [int(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]*\b", quote)]
        po_total = int(order["qty_units"])
        split_context = (
            q == po_total and any(n < po_total for n in numbers)
            and ("balance" in quote.lower() or "first lot" in quote.lower()
                 or "ippo illa" in quote.lower() or "aagum" in quote.lower())
        )
        if split_context:
            return "source describes a partial/future lot and remaining balance; it does not support full completion"
    if event.get("event_status") == "completed" and re.search(r"\b(will|would|going to|planned|expect(?:ed)?)\b", quote, re.I):
        return "source states future intent, not confirmed completion"
    return None


def _source_supports_event(quote, event):
    """Require a source assertion appropriate to the event, not merely matching numbers."""
    text = _norm(quote).lower()
    status, milestone, kind = event.get("event_status"), event.get("milestone"), event.get("event_type")
    if milestone == "approval":
        return any(term in text for term in ("approval", "approved", "approval done"))
    if milestone == "trims_ready":
        if status == "incomplete":
            return "trims not received" in text or "trims have not arrived" in text
        if status == "completed":
            return "trims received" in text or "trims have arrived" in text
    if milestone == "fabric_ready":
        return "fabric received" in text or "fabric has arrived" in text
    if milestone in ("cutting", "cutting_start"):
        if kind == "cutting_start" and status in ("in_progress", "completed"):
            return "cutting started" in text or "cutting start" in text
        if kind == "cutting_progress" and status == "in_progress":
            return "cut pieces" in text or "pieces cut" in text
        if kind == "cutting_done" and status == "completed":
            return any(term in text for term in ("cutting completed", "cutting complete", "fully cut"))
    if milestone == "sewing_start" and kind == "sewing_start":
        return status == "in_progress" and any(term in text for term in ("sewing started", "stitching started"))
    if milestone == "packing" and kind == "packing_done":
        return status == "completed" and any(term in text for term in ("packing completed", "packing done"))
    if milestone == "qc" and kind == "qc_result":
        return any(term in text for term in ("passed", "checked", "reject"))
    return False


def _source_supports_event(quote, event):
    """Require a source assertion appropriate to the event, not merely matching numbers."""
    text = _norm(quote).lower()
    status, milestone, kind = event.get("event_status"), event.get("milestone"), event.get("event_type")
    if milestone == "approval":
        return any(term in text for term in ("approval", "approved", "approval done"))
    if milestone == "trims_ready":
        if status == "incomplete":
            return "trims not received" in text or "trims have not arrived" in text
        if status == "completed":
            return "trims received" in text or "trims have arrived" in text
    if milestone == "fabric_ready":
        return "fabric received" in text or "fabric has arrived" in text
    if milestone in ("cutting", "cutting_start"):
        if kind == "cutting_start" and status in ("in_progress", "completed"):
            return "cutting started" in text or "cutting start" in text
        if kind == "cutting_progress" and status == "in_progress":
            return "cut pieces" in text or "pieces cut" in text
        if kind == "cutting_done" and status == "completed":
            return any(term in text for term in ("cutting completed", "cutting complete", "fully cut"))
    if milestone == "sewing_start" and kind == "sewing_start":
        return status == "in_progress" and any(term in text for term in ("sewing started", "stitching started"))
    if milestone == "packing" and kind == "packing_done":
        return status == "completed" and any(term in text for term in ("packing completed", "packing done"))
    if milestone == "qc" and kind == "qc_result":
        return any(term in text for term in ("passed", "checked", "reject"))
    return False


def s1_guardrails(events, orders, source_records):
    """Accept only typed events whose PO and exact quote resolve to the cited source record."""
    sources = {r["source_id"]: r for r in source_records}
    checked, accepted, held = [], [], []
    for event in events:
        item = dict(event)
        notes = []
        if item.get("milestone") in ("cutting_done", "cutting_progress"):
            item["milestone"] = "cutting"
        source = sources.get(event.get("source_id"))
        po = event.get("po_id")
        if po not in orders:
            notes.append("unknown PO id")
        if event.get("event_type") not in EVENT_TYPES:
            notes.append("unknown event_type")
        if event.get("event_status") not in EVENT_STATUSES:
            notes.append("missing or invalid event_status")
        if event.get("milestone") not in MILESTONES:
            notes.append("missing or invalid milestone")
        if not source:
            notes.append("unknown or missing source_id")
        elif po not in source["po_ids"]:
            notes.append("source record does not identify this PO")
        elif not event.get("source_quote") or _norm(event["source_quote"]) not in _norm(source["text"]):
            notes.append("source_quote is not an exact excerpt of cited source")
        elif not _source_supports_event(event.get("source_quote", ""), event):
            notes.append("source quote does not explicitly support this event type, milestone, and status")
        elif not _source_supports_event(event.get("source_quote", ""), event):
            notes.append("source quote does not explicitly support this event type, milestone, and status")
        if source and source.get("language") == "ta-en" and event.get("event_status") in ("completed", "in_progress", "incomplete"):
            notes.append("mixed-language status requires human confirmation")
        q = event.get("quantity")
        if q is not None and (not isinstance(q, (int, float)) or isinstance(q, bool) or q < 0):
            notes.append("invalid quantity")
        if q is not None and po in orders and q > int(orders[po]["qty_units"]):
            notes.append("quantity exceeds PO size")
        if q is not None and event.get("quantity_basis") not in ("reported_actual", "planned_target", "order_total"):
            notes.append("quantity basis is unknown")
        if event.get("event_status") == "future_intent":
            notes.append("future intent is not an actual event")
        if event.get("event_status") == "unknown" or event.get("needs_human_review"):
            notes.append("event is uncertain or explicitly marked for review")
        raw_confidence = event.get("confidence")
        confidence_labels = {"high": 0.9, "medium": 0.6, "low": 0.3}
        if isinstance(raw_confidence, str) and raw_confidence.lower() in confidence_labels:
            confidence_score = confidence_labels[raw_confidence.lower()]
        elif isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
            confidence_score = float(raw_confidence)
        else:
            confidence_score = None
            notes.append("missing or invalid model confidence")
        item["confidence_score"] = confidence_score
        if confidence_score is not None and confidence_score < 0.75:
            notes.append("low model confidence")
        if source and po in orders:
            item["reported_at"] = source["observed_at"]
            item["source_language"] = source["language"]
            contradiction = _future_or_split_quote(event.get("source_quote", ""), event, orders[po])
            if contradiction:
                notes.append(contradiction)
        event_time = event.get("occurred_at")
        if event_time:
            try:
                event_dt = dt.datetime.fromisoformat(event_time)
            except (TypeError, ValueError):
                notes.append("invalid occurred_at")
            else:
                if source:
                    report_dt = dt.datetime.fromisoformat(source["observed_at"])
                    if event_dt.tzinfo is None:
                        event_dt = dt.datetime.combine(event_dt.date(), dt.time.min, tzinfo=report_dt.tzinfo)
                    if event_dt > report_dt:
                        notes.append("event occurrence is later than source report time")
                if source and _norm(event_time) not in _norm(source["text"]):
                    notes.append("occurred_at is not stated in source quote")
        item["validation_status"] = "held" if notes else "accepted"
        item["guardrail_notes"] = sorted(set(notes))
        checked.append(item)

    # Same-time incompatible updates remain together in review; confidence does not break ties.
    groups = {}
    for item in checked:
        source = sources.get(item.get("source_id"))
        if (not source or item.get("po_id") not in orders
                or _norm(item.get("source_quote", "")) not in _norm(source["text"])
                or not _source_supports_event(item.get("source_quote", ""), item)):
            continue
        key = (item.get("po_id"), item.get("milestone"), item.get("occurred_at") or
               (sources.get(item.get("source_id")) or {}).get("observed_at"))
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        statuses = {i.get("event_status") for i in group}
        quantities = {i.get("quantity") for i in group if i.get("quantity") is not None}
        if len(group) > 1 and (len(statuses) > 1 or len(quantities) > 1):
            for item in group:
                item["validation_status"] = "held"
                item["guardrail_notes"].append("conflicting same-time updates for the same PO milestone")
    for item in checked:
        (accepted if item["validation_status"] == "accepted" else held).append(item)
    return {"accepted": accepted, "flagged": held, "all": checked}


def _event_id(event):
    return f"{event['source_id']}:{event.get('po_id')}:{event.get('milestone')}:{event.get('event_status')}"


def render_draft(candidate, orders, events, sources, as_of):
    po = candidate["po_id"]
    row = orders[po]
    evidence = candidate["evidence"]
    event_refs = [x for x in evidence if x["kind"] == "event"]
    plan_refs = [x for x in evidence if x["kind"] == "plan"]
    if candidate["draft_kind"] == "progress_check":
        ev = next(e for e in events if _event_id(e) == event_refs[0]["event_id"])
        q = ev.get("quantity")
        total = int(row["qty_units"])
        q_text = f"{int(q):,}" if isinstance(q, (int, float)) and float(q).is_integer() else str(q)
        return (f"{po}: source {ev['source_id']} reports {q_text} cut pieces today; the plan quantity is "
                f"{total:,}. This is a progress report, not confirmation that the PO is complete. "
                "Impact to later milestones and ship date is unknown. "
                "Please review the source notes and confirm the current cutting status.")
    if candidate["draft_kind"] == "same_day_status_check":
        ev = next(e for e in events if _event_id(e) == event_refs[0]["event_id"])
        plan = next(p for p in plan_refs if p["field"] == "trims_ready_plan")
        source = sources[ev["source_id"]]
        return (f"{po}: {ev['source_id']} reported trims not received at {source['observed_at'][11:16]}; "
                f"the planned trims-ready date is {plan['value']}. The report and plan are on the same date, "
                "so a delay is not established; receipt after the report time is unknown. "
                "Impact to sewing and shipment is unknown. Please confirm the current receipt status.")
    if candidate["draft_kind"] == "overdue_incomplete":
        ev = next(e for e in events if _event_id(e) == event_refs[0]["event_id"])
        plan = plan_refs[0]
        overdue_days = (_date(as_of) - _date(plan["value"])).days
        return (f"{po}: source {ev['source_id']} reports {ev['milestone']} remains incomplete as of "
                f"{ev['reported_at']}; the planned date was {plan['value']}, {overdue_days} calendar days earlier. "
                "This establishes an overdue incomplete milestone only. Impact to later milestones and shipment is unknown. "
                "Please review and confirm the current status.")
    if candidate["draft_kind"] == "late_completion_record":
        ev = next(e for e in events if _event_id(e) == event_refs[0]["event_id"])
        plan = plan_refs[0]
        actual = ev["occurred_at"][:10]
        late_days = (_date(actual) - _date(plan["value"])).days
        return (f"{po}: source {ev['source_id']} reports {ev['milestone']} completed on {actual}; "
                f"the planned date was {plan['value']} ({late_days} calendar days earlier). "
                "Impact to later milestones and shipment is unknown.")
    raise ValueError("unknown deterministic draft kind")


def build_alerts(orders, accepted_events, flagged_events, sources, as_of):
    """Create only deterministic, source-linked progress/status candidates; never infer impacts."""
    sources = {r["source_id"]: r for r in sources}
    drafts = []
    for event in accepted_events:
        source = sources.get(event.get("source_id"))
        if source and dt.datetime.fromisoformat(source["observed_at"]) > dt.datetime.fromisoformat(as_of):
            continue
        status = event.get("event_status")
        po = event["po_id"]
        milestone = event.get("milestone")
        plan_field = PLAN_FIELDS.get(milestone)
        plan_value = orders[po].get(plan_field) if plan_field else None
        event_ref = {"kind": "event", "event_id": _event_id(event), "source_id": event["source_id"],
                     "source_quote": event["source_quote"], "status": status,
                     "quantity": event.get("quantity"), "occurred_at": event.get("occurred_at")}
        candidate = None
        if status == "in_progress" and event.get("quantity") is not None and milestone in ("cutting", "cutting_start", "cutting_done"):
            candidate = {"draft_kind": "progress_check", "po_id": po, "evidence": [event_ref,
                {"kind": "plan", "po_id": po, "field": "qty_units", "value": orders[po]["qty_units"]}]}
        elif status == "incomplete" and plan_value and milestone == "trims_ready":
            delta = (_date(as_of) - _date(plan_value)).days
            if delta == 0:
                candidate = {"draft_kind": "same_day_status_check", "po_id": po, "evidence": [event_ref,
                    {"kind": "plan", "po_id": po, "field": plan_field, "value": plan_value}]}
            elif delta > 0:
                candidate = {"draft_kind": "overdue_incomplete", "po_id": po, "evidence": [event_ref,
                    {"kind": "plan", "po_id": po, "field": plan_field, "value": plan_value}]}
        elif status == "completed" and event.get("occurred_at") and plan_value:
            actual_date = event["occurred_at"][:10]
            if _date(actual_date) > _date(plan_value):
                candidate = {"draft_kind": "late_completion_record", "po_id": po, "evidence": [event_ref,
                    {"kind": "plan", "po_id": po, "field": plan_field, "value": plan_value}]}
        if candidate:
            candidate["candidate_id"] = f"{po}-{candidate['draft_kind']}-{event['source_id']}"
            candidate["validation_status"] = "accepted_for_human_review"
            candidate["human_decision"] = "undecided"
            candidate["suggested_action"] = None
            candidate["suggested_owner"] = None
            candidate["draft_message"] = render_draft(candidate, orders, accepted_events, sources, as_of)
            drafts.append(candidate)
    return drafts


def s2_guardrails(alerts, orders, accepted_events, source_records, as_of):
    """Verify typed evidence references and deterministic rendering; prose is never model-generated."""
    sources = {r["source_id"]: r for r in source_records}
    event_map = {_event_id(e): e for e in accepted_events}
    ok, held = [], []
    for original in alerts:
        item = dict(original)
        notes = []
        if item.get("po_id") not in orders:
            notes.append("unknown PO id")
        if item.get("draft_kind") not in ("progress_check", "same_day_status_check", "overdue_incomplete", "late_completion_record"):
            notes.append("missing/unsupported deterministic draft kind")
        if not isinstance(item.get("evidence"), list) or not item["evidence"]:
            notes.append("missing structured evidence references")
        else:
            event_refs = [x for x in item["evidence"] if isinstance(x, dict) and x.get("kind") == "event"]
            plan_refs = [x for x in item["evidence"] if isinstance(x, dict) and x.get("kind") == "plan"]
            for ref in item["evidence"]:
                if not isinstance(ref, dict):
                    notes.append("unstructured evidence reference")
                    continue
                if ref.get("kind") == "event":
                    event = event_map.get(ref.get("event_id"))
                    source = sources.get(ref.get("source_id"))
                    if not event or not source or event.get("source_quote") != ref.get("source_quote"):
                        notes.append("event evidence reference does not resolve to an accepted source event")
                    elif event.get("po_id") != item.get("po_id") or event.get("source_id") != ref.get("source_id"):
                        notes.append("event evidence PO/source mismatch")
                elif ref.get("kind") == "plan":
                    po, field = ref.get("po_id"), ref.get("field")
                    if po != item.get("po_id") or po not in orders or orders[po].get(field) != str(ref.get("value")):
                        notes.append("plan evidence does not match this PO row and field")
                else:
                    notes.append("unknown evidence reference kind")
            if len(event_refs) != 1:
                notes.append("draft must reference exactly one accepted event")
            if not plan_refs:
                notes.append("draft must cite a matching plan fact")
            if len(event_refs) == 1:
                event = event_map.get(event_refs[0].get("event_id"))
                kind = item.get("draft_kind")
                if kind == "progress_check" and not (event and event.get("event_status") == "in_progress"
                        and event.get("milestone") in ("cutting", "cutting_start", "cutting_done")
                        and event.get("quantity") is not None
                        and any(p.get("field") == "qty_units" for p in plan_refs)):
                    notes.append("progress draft does not match a quantified in-progress event and PO quantity")
                if kind in ("same_day_status_check", "overdue_incomplete"):
                    if not (event and event.get("event_status") == "incomplete"
                            and event.get("milestone") == "trims_ready"
                            and any(p.get("field") == "trims_ready_plan" for p in plan_refs)):
                        notes.append("trims status draft lacks an explicit incomplete trims event and its plan date")
                    else:
                        plan = next(p for p in plan_refs if p.get("field") == "trims_ready_plan")
                        days = (_date(as_of) - _date(plan["value"])).days
                        if (kind == "same_day_status_check" and days != 0) or (kind == "overdue_incomplete" and days <= 0):
                            notes.append("draft kind does not match as-of date versus planned date")
                if kind == "late_completion_record":
                    if not (event and event.get("event_status") == "completed" and event.get("occurred_at")):
                        notes.append("late completion draft lacks explicit completed status and occurrence time")
                    elif plan_refs:
                        days = (_date(event["occurred_at"]) - _date(plan_refs[0]["value"])).days
                        if days <= 0:
                            notes.append("completion was not after its plan date")
        if item.get("suggested_action") is not None or item.get("suggested_owner") is not None:
            notes.append("actions and owners must not be invented")
        if not notes:
            try:
                expected = render_draft(item, orders, accepted_events, sources, as_of)
            except (KeyError, StopIteration, ValueError):
                notes.append("draft cannot be rendered from cited facts")
            else:
                if item.get("draft_message") != expected:
                    notes.append("draft text differs from deterministic source-linked rendering")
        item["validation_status"] = "held" if notes else "accepted_for_human_review"
        item["guardrail_notes"] = notes
        (held if notes else ok).append(item)
    return {"accepted": ok, "flagged": held}


def write_drafts_log(path, as_of, events, held_events, drafts, held_drafts):
    path.write_text(json.dumps({"as_of": as_of, "events": events, "held_events": held_events,
                                "drafts": drafts, "held_drafts": held_drafts}, indent=2, ensure_ascii=False) + "\n")


def write_outbox(run_id, approved, unresolved_holds):
    """Only called after every draft has a human decision; only approved decisions go here."""
    outdir = ROOT / "outbox"
    outdir.mkdir(exist_ok=True)
    data = {"run_id": run_id, "approved": approved, "held_for_review": unresolved_holds}
    (outdir / f"escalations-{run_id}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    md = [f"# Human-approved drafts for {run_id}\n"]
    for item in approved:
        md.extend([f"## {item['po_id']}\n", item["draft_message"] + "\n"])
    (outdir / f"escalations-{run_id}.md").write_text("\n".join(md))
