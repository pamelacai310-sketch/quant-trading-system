#!/usr/bin/env bash
set -uo pipefail

REQUIRED_MODULES="${QTS_REQUIRED_PY_MODULES:-pandas numpy akshare yfinance}"

python_has_modules() {
  local candidate="$1"
  [[ -x "${candidate}" ]] || return 1
  "${candidate}" - "${REQUIRED_MODULES}" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

required = sys.argv[1].split()
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(1)
PY
}

emit_if_valid() {
  local candidate="$1"
  if [[ -n "${candidate}" ]] && python_has_modules "${candidate}"; then
    echo "${candidate}"
    exit 0
  fi
}

emit_if_valid "${QTS_PYTHON_BIN:-}"
emit_if_valid "/Applications/Xcode.app/Contents/Developer/usr/bin/python3"
emit_if_valid "/usr/bin/python3"
emit_if_valid "$(command -v python3 || true)"
emit_if_valid "/opt/homebrew/bin/python3"
emit_if_valid "/usr/local/bin/python3"

echo "No Python interpreter found with required modules: ${REQUIRED_MODULES}" >&2
exit 1
