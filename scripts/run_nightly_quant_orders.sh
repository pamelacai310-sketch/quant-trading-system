#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/state/nightly_logs"
STATUS_DIR="${REPO_ROOT}/state/nightly_status"
REPORT_DIR="${REPO_ROOT}/state/nightly_reports"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "${REPORT_DIR}"

_china_date() {
  TZ=Asia/Shanghai date "$@"
}

_yesterday() {
  if TZ=Asia/Shanghai date -v-1d +%F >/dev/null 2>&1; then
    TZ=Asia/Shanghai date -v-1d +%F
  else
    TZ=Asia/Shanghai date -d yesterday +%F
  fi
}

NOW_HOUR="$(_china_date +%H)"
if [[ -n "${QTS_NIGHTLY_DATE:-}" ]]; then
  RUN_DATE="${QTS_NIGHTLY_DATE}"
elif [[ "${QTS_NIGHTLY_CATCH_UP:-1}" == "1" && $((10#${NOW_HOUR})) -lt 4 ]]; then
  RUN_DATE="$(_yesterday)"
else
  RUN_DATE="$(_china_date +%F)"
fi

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/nightly_quant_orders_${RUN_DATE}_${STAMP}.log"
STATUS_FILE="${STATUS_DIR}/nightly_quant_orders_${RUN_DATE}.json"
LATEST_STATUS="${STATUS_DIR}/latest.json"
REPORT_FILE="${REPORT_DIR}/${RUN_DATE}.json"
LOCK_DIR="${LOG_DIR}/nightly_quant_orders.lock"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export QTS_FUTURES_SETTLE_FALLBACK="${QTS_FUTURES_SETTLE_FALLBACK:-daily_main_contract}"
export TZ="${TZ:-Asia/Shanghai}"
export PATH="${QTS_NIGHTLY_PATH:-/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"

PYTHON_BIN="$(bash "${SCRIPT_DIR}/resolve_nightly_python.sh")"
CAFFEINATE_BIN="${QTS_CAFFEINATE_BIN:-$(command -v caffeinate || true)}"
NIGHTLY_TIMEOUT_SECONDS="${QTS_NIGHTLY_TIMEOUT_SECONDS:-3600}"

write_status() {
  local status="$1"
  local reason="$2"
  local run_exit="${3:-0}"
  local health_exit="${4:-0}"
  local finished_at
  finished_at="$(_china_date -Iseconds)"
  "${PYTHON_BIN}" - "${STATUS_FILE}" "${LATEST_STATUS}" <<PY
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
latest_path = Path(sys.argv[2])
payload = {
    "status": "${status}",
    "reason": "${reason}",
    "run_date": "${RUN_DATE}",
    "started_at": "${STARTED_AT:-}",
    "finished_at": "${finished_at}",
    "run_exit": int("${run_exit}"),
    "health_exit": int("${health_exit}"),
    "repo_root": "${REPO_ROOT}",
    "log_file": "${LOG_FILE}",
    "report_file": "${REPORT_FILE}",
    "python": "${PYTHON_BIN}",
}
status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

if [[ -z "${QTS_NIGHTLY_DATE:-}" && "${QTS_NIGHTLY_FORCE:-0}" != "1" && $((10#${NOW_HOUR})) -lt 20 && $((10#${NOW_HOUR})) -ge 4 ]]; then
  STARTED_AT="$(_china_date -Iseconds)"
  {
    echo "started_at=${STARTED_AT}"
    echo "repo_root=${REPO_ROOT}"
    echo "run_date=${RUN_DATE}"
    echo "skip_reason=before_20_00_asia_shanghai"
  } >>"${LOG_FILE}" 2>&1
  write_status "skipped" "before_20_00_asia_shanghai" 0 0
  exit 0
fi

if [[ -z "${QTS_NIGHTLY_DATE:-}" && "${QTS_NIGHTLY_FORCE:-0}" != "1" && "${QTS_NIGHTLY_SKIP_EXISTING:-1}" == "1" && -s "${REPORT_FILE}" ]]; then
  STARTED_AT="$(_china_date -Iseconds)"
  {
    echo "started_at=${STARTED_AT}"
    echo "repo_root=${REPO_ROOT}"
    echo "run_date=${RUN_DATE}"
    echo "skip_reason=target_report_already_exists"
    echo "report_file=${REPORT_FILE}"
  } >>"${LOG_FILE}" 2>&1
  write_status "skipped" "target_report_already_exists" 0 0
  exit 0
fi

if ! mkdir "${LOCK_DIR}" >/dev/null 2>&1; then
  STARTED_AT="$(_china_date -Iseconds)"
  {
    echo "started_at=${STARTED_AT}"
    echo "repo_root=${REPO_ROOT}"
    echo "run_date=${RUN_DATE}"
    echo "skip_reason=nightly_job_already_running"
    echo "lock_dir=${LOCK_DIR}"
  } >>"${LOG_FILE}" 2>&1
  write_status "skipped" "nightly_job_already_running" 0 0
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true' EXIT

STARTED_AT="$(TZ=Asia/Shanghai date -Iseconds)"
run_exit=0
{
  echo "started_at=${STARTED_AT}"
  echo "repo_root=${REPO_ROOT}"
  echo "run_date=${RUN_DATE}"
  echo "python=${PYTHON_BIN}"
  echo "caffeinate=${CAFFEINATE_BIN:-unavailable}"
  echo "timeout_seconds=${NIGHTLY_TIMEOUT_SECONDS}"
  cd "${REPO_ROOT}"
  run_cmd=("${PYTHON_BIN}" -m quant_trade_system.nightly_quant_orders --date "${RUN_DATE}")
  if [[ -n "${CAFFEINATE_BIN}" && "${QTS_NIGHTLY_PREVENT_SLEEP:-1}" == "1" ]]; then
    run_cmd=("${CAFFEINATE_BIN}" -dimsu "${run_cmd[@]}")
  fi
  if [[ "${NIGHTLY_TIMEOUT_SECONDS}" =~ ^[0-9]+$ && "${NIGHTLY_TIMEOUT_SECONDS}" -gt 0 ]]; then
    "${PYTHON_BIN}" - "${NIGHTLY_TIMEOUT_SECONDS}" "${run_cmd[@]}" <<'PY'
import os
import signal
import subprocess
import sys

timeout = int(sys.argv[1])
cmd = sys.argv[2:]
proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
try:
    raise SystemExit(proc.wait(timeout=timeout))
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(f"nightly_timeout={timeout}", file=sys.stderr)
    raise SystemExit(124)
PY
  else
    "${run_cmd[@]}"
  fi
  run_exit=$?
  echo "nightly_exit=${run_exit}"
  echo "finished_at=$(_china_date -Iseconds)"
} >>"${LOG_FILE}" 2>&1

if [[ "${run_exit}" -ne 0 ]]; then
  write_status "failed" "nightly_runner_failed" "${run_exit}" 0
fi

health_exit=0
"${PYTHON_BIN}" -m quant_trade_system.nightly_health_check --date "${RUN_DATE}" --repo-root "${REPO_ROOT}" --json >>"${LOG_FILE}" 2>&1
health_exit=$?
echo "health_exit=${health_exit}" >>"${LOG_FILE}" 2>&1

if [[ "${run_exit}" -ne 0 ]]; then
  write_status "failed" "nightly_runner_failed" "${run_exit}" "${health_exit}"
  exit "${run_exit}"
fi

if [[ "${health_exit}" -ne 0 ]]; then
  write_status "failed" "nightly_health_check_failed" "${run_exit}" "${health_exit}"
  exit "${health_exit}"
fi

write_status "ok" "nightly_completed" "${run_exit}" "${health_exit}"
exit 0
