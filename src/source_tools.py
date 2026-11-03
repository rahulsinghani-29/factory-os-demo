"""Read-only fixture adapters at the future factory-API boundaries. No HTTP calls."""
from copy import deepcopy


def read_fixture(records, resource, scope):
    value = records.get(resource)
    if resource == 'po_details_api' and isinstance(value, dict):
        requested = scope.get('po_id')
        recorded = value.get('po_id')
        if requested and recorded and requested != recorded.upper():
            raise ValueError('No matching PO in the supplied fixture snapshot')
    if resource == 'plan_api' and isinstance(value, dict):
        requested = scope.get('line')
        recorded = value.get('line')
        if requested and recorded and requested != recorded.upper():
            raise ValueError('No matching line in the supplied plan snapshot')
    # Missing records remain missing; do not manufacture confirmations or API success.
    return deepcopy(value)
