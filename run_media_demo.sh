#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -z "${PYTHON:-}" ] && [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON=${PYTHON:-python3.11}
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.11+ is required for Streamlit. Set PYTHON to its executable." >&2
  exit 2
fi

if ! "$PYTHON" -c 'import streamlit' >/dev/null 2>&1; then
  echo "Streamlit is not installed for $PYTHON. Create a venv and install requirements-media.txt; this script does not install packages." >&2
  exit 2
fi

cd "$ROOT"
export QUINCE_OCR_DET_MODEL_DIR=${QUINCE_OCR_DET_MODEL_DIR:-"$ROOT/models/PP-OCRv5_mobile_det_infer"}
export QUINCE_OCR_TA_MODEL_DIR=${QUINCE_OCR_TA_MODEL_DIR:-"$ROOT/models/ta_PP-OCRv5_mobile_rec_infer"}
exec "$PYTHON" -m streamlit run app.py "$@"
