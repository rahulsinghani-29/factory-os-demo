"""Draft labelled reference cases, deliberately separate from inference inputs."""
import csv
import io
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'data' / 'golden_cases.json'
LABEL_FIELDS = {
    'po_id / line / as_of': 'Expected identity and explicit event time, including timezone.',
    'event_state': 'Future plan, reported partial work, reported completion, or unclear. Reported is not verified receipt.',
    'hourly_quantities / total_reported_sewn': 'Incremental counts. Future blank hours are null, not zero.',
    'expected_outcome / expected_delay_risk': 'Clear or unclear; yes, no or null. Scope is sewing risk, not shipment readiness.',
    'expected_route / expected_reason': 'Expected terminal route and the reason for it.',
    'qc_completion': 'Actual QC confirmation status; missing evidence stays unknown.',
    'evidence_ids / rationale': 'References and the reason an independent reviewer should agree with the label.',
    'forbidden_claims': 'Unsupported statements that must fail a run even if its aggregate judge score is high.',
    'review_status / known_implementation_gap': 'Whether a person reviewed the reference, and whether an existing bug is represented.',
    'source_id / source_kind / language / source_asset': 'Input provenance. A missing media asset means no perception benchmark.',
    'reference_text / context_snapshot': 'Exact input plus PO due date, requirement, plan, handoff/QC and prior-message snapshots.',
}


def load_golden():
    return json.loads(PATH.read_text())


def golden_rows(data=None):
    return [{'Case': c['case_id'], 'Input / issue': c['title'], 'Input excerpt': c['reference_text'].splitlines()[-1], 'Kind': c['source_kind'],
             'PO / line': c['labels']['po_id']+' / '+c['labels']['line'],
             'Reported sewn': c['labels']['total_reported_sewn'], 'State': c['labels']['event_state'],
             'Expected result': c['labels']['expected_outcome'],
             'Sewing risk': c['labels']['expected_delay_risk'] or 'unknown',
             'Reason': c['labels']['expected_reason'],
             'Known gap': c['labels']['known_implementation_gap']}
            for c in (data or load_golden())['cases']]


def golden_csv(data=None):
    rows = golden_rows(data)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
