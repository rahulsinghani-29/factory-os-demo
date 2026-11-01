"""Opt-in, synthetic-only tracing for the local LangGraph demo."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
import time

_ACTIVE_CLIENT = ContextVar("quince_langfuse_client", default=None)
_ACTIVE_OBSERVATION = ContextVar("quince_langfuse_observation", default=None)
_ACTIVE_OBSERVATION_RECORDED = ContextVar("quince_langfuse_observation_recorded", default=False)
_LOG = logging.getLogger(__name__)

_NODE_TYPES = {
    "route_input": "chain",
    "transcribe_audio": "tool",
    "ocr_photo": "tool",
    "extract_candidates": "chain",
    "read_matching_records": "retriever",
    "reconcile": "evaluator",
    "revalidate_after_review": "guardrail",
    "idempotent_demo_commit": "tool",
    "no_op": "event",
}

_SAFE_NODE_FIELDS = {
    "route_input": {"modality"},
    "transcribe_audio": {"engine", "segment_count", "segment_timing"},
    "ocr_photo": {"engine", "block_count", "row_count"},
    "extract_candidates": {"model", "endpoint", "evidence_ids", "candidate_count"},
    "read_matching_records": {"receipt_count", "qc_count", "ledger_count"},
    "reconcile": {"candidate_routes", "match_ids", "confirmed_receipt_total"},
    "revalidate_after_review": {"decisions"},
    "idempotent_demo_commit": {"write_count", "confirmed_receipt_total_after"},
    "no_op": {"status", "confirmed_receipt_total", "count_change"},
}


def langfuse_configured():
    """Return configuration presence only; never read or print key values."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and
                os.environ.get("LANGFUSE_SECRET_KEY") and
                os.environ.get("LANGFUSE_BASE_URL"))


def langfuse_enabled_for(synthetic, user_opt_in):
    return bool(synthetic and user_opt_in and langfuse_configured())


def langfuse_auth_check():
    """Check cloud authentication without exposing credentials or creating a trace."""
    if not langfuse_configured():
        return False
    try:
        from langfuse import get_client
        return bool(get_client().auth_check())
    except Exception as exc:
        _LOG.warning("Langfuse authentication check failed (%s)", type(exc).__name__)
        return False


@contextmanager
def langfuse_run(synthetic, user_opt_in, run_id, modality, source_id=None, thread_id=None):
    """Create one root trace only after synthetic-data opt-in and local config."""
    if not langfuse_enabled_for(synthetic, user_opt_in):
        yield None
        return
    try:
        from langfuse import get_client, propagate_attributes
        client = get_client()
    except Exception as exc:
        _LOG.warning("Langfuse setup failed (%s); continuing with local traces only", type(exc).__name__)
        yield None
        return
    try:
        with client.start_as_current_observation(
            as_type="chain", name="quince-factory-os-run",
            input={"run_id": run_id, "source_id": source_id,
                   "modality": modality, "synthetic": True},
            metadata={"project": "prototype-Q", "synthetic_only": True},
        ) as root:
            token = _ACTIVE_CLIENT.set(client)
            try:
                with propagate_attributes(
                    session_id=thread_id,
                    tags=["prototype-Q", "synthetic", "media-demo"],
                    metadata={"run_id": run_id, "source_id": source_id,
                              "modality": modality, "synthetic_only": True},
                    trace_name="quince-factory-os-run",
                    environment="development",
                ):
                    yield root
            finally:
                _ACTIVE_CLIENT.reset(token)
    except Exception as exc:
        _LOG.warning("Langfuse trace failed during the workflow (%s)", type(exc).__name__)
        raise
    finally:
        try:
            client.flush()
        except Exception as exc:
            _LOG.warning("Langfuse flush failed (%s); local traces remain available", type(exc).__name__)


@contextmanager
def langfuse_generation(name, model, input_data):
    client = _ACTIVE_CLIENT.get()
    if client is None:
        yield None
        return
    try:
        context = client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input_data,
        )
        observation = context.__enter__()
    except Exception as exc:
        _LOG.warning("Langfuse generation setup failed (%s)", type(exc).__name__)
        yield None
        return
    try:
        yield observation
    except Exception as exc:
        try:
            observation.update(level="ERROR", status_message=type(exc).__name__)
        except Exception:
            pass
        context.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        context.__exit__(None, None, None)


def record_node(node, status, latency_seconds, fields):
    """Update the active, typed graph node with a data-minimized result."""
    observation = _ACTIVE_OBSERVATION.get()
    if observation is None:
        return
    fields = fields or {}
    allowed = _SAFE_NODE_FIELDS.get(node, set())
    safe_output = {key: fields[key] for key in allowed if key in fields}
    if node == "read_matching_records":
        safe_output = {"receipt_count": len(fields.get("receipt_event_ids", [])),
                       "qc_count": len(fields.get("qc_event_ids", [])),
                       "ledger_count": len(fields.get("ledger_event_ids", []))}
    elif node == "idempotent_demo_commit":
        safe_output = {"write_count": sum(bool(item.get("written"))
                                          for item in fields.get("results", [])
                                          if isinstance(item, dict)),
                       "confirmed_receipt_total_after": fields.get("confirmed_receipt_total_after")}
    elif node == "revalidate_after_review":
        decisions = [item for item in fields.get("decisions", []) if isinstance(item, dict)]
        safe_output = {"decision_count": len(decisions),
                       "decision_types": [item.get("decision") for item in decisions],
                       "write_kinds": [item.get("write_kind") for item in decisions],
                       "all_valid": all(item.get("valid") is True for item in decisions)}
    if "error" in fields:
        safe_output["error_type"] = str(fields["error"]).split(":", 1)[0][:80]
    try:
        observation.update(output={"status": status, **safe_output},
                           level="ERROR" if status == "failed" else "DEFAULT",
                           metadata={"latency_seconds": round(latency_seconds, 4)})
        _ACTIVE_OBSERVATION_RECORDED.set(True)
    except Exception as exc:
        _LOG.warning("Langfuse node observation failed (%s)", type(exc).__name__)


@contextmanager
def langfuse_node(node, state):
    """Trace one graph node for its full duration with a minimal input summary."""
    client = _ACTIVE_CLIENT.get()
    if client is None:
        yield None
        return
    source = state.get("source") or {}
    input_summary = {"modality": state.get("modality"),
                     "source_id": source.get("source_id"),
                     "candidate_count": len(state.get("candidates", [])),
                     "record_count": len((state.get("fixture") or {}).get("confirmed_receipts", []))}
    try:
        context = client.start_as_current_observation(
            as_type=_NODE_TYPES.get(node, "span"), name=node,
            input=input_summary,
        )
        observation = context.__enter__()
    except Exception as exc:
        _LOG.warning("Langfuse node setup failed (%s)", type(exc).__name__)
        yield None
        return
    token = _ACTIVE_OBSERVATION.set(observation)
    recorded_token = _ACTIVE_OBSERVATION_RECORDED.set(False)
    started = time.monotonic()
    try:
        yield observation
    except BaseException as exc:
        if type(exc).__name__ == "GraphInterrupt":
            try:
                observation.update(output={"status": "interrupted"},
                                   metadata={"latency_seconds": round(time.monotonic() - started, 4)})
                context.__exit__(None, None, None)
            except Exception as telemetry_exc:
                _LOG.warning("Langfuse interrupt span close failed (%s)", type(telemetry_exc).__name__)
        else:
            try:
                observation.update(level="ERROR", status_message=type(exc).__name__,
                                   metadata={"latency_seconds": round(time.monotonic() - started, 4)})
                context.__exit__(type(exc), exc, exc.__traceback__)
            except Exception as telemetry_exc:
                _LOG.warning("Langfuse failed node close (%s)", type(telemetry_exc).__name__)
        raise
    else:
        try:
            if not _ACTIVE_OBSERVATION_RECORDED.get():
                observation.update(output={"status": "completed"})
            observation.update(metadata={"latency_seconds": round(time.monotonic() - started, 4)})
            context.__exit__(None, None, None)
        except Exception as exc:
            _LOG.warning("Langfuse node close failed (%s)", type(exc).__name__)
    finally:
        _ACTIVE_OBSERVATION_RECORDED.reset(recorded_token)
        _ACTIVE_OBSERVATION.reset(token)


def update_run_output(root, result):
    """Add a small outcome summary to the root observation, without source text."""
    if root is None:
        return
    reconciliations = result.get("reconciliations", []) if isinstance(result, dict) else []
    final_result = result.get("result", {}) if isinstance(result, dict) else {}
    payload = {"status": final_result.get("status", "completed"),
               "claim_count": len(reconciliations),
               "routes": [item.get("route") for item in reconciliations if isinstance(item, dict)],
               "human_review": "not_required",
               "human_operational_approval": False,
               "write_count": 0,
               "evaluation_sampling": "not_configured"}
    try:
        root.update(output=payload)
    except Exception as exc:
        _LOG.warning("Langfuse root output failed (%s)", type(exc).__name__)
