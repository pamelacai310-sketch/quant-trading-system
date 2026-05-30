#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/state/nightly_logs"
mkdir -p "${LOG_DIR}"

RUN_DATE="${QTS_NIGHTLY_DATE:-$(TZ=Asia/Shanghai date +%F)}"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/nightly_quant_orders_${RUN_DATE}_${STAMP}.log"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export QTS_FUTURES_SETTLE_FALLBACK="${QTS_FUTURES_SETTLE_FALLBACK:-daily_main_contract}"
export TZ="${TZ:-Asia/Shanghai}"

{
  echo "started_at=$(TZ=Asia/Shanghai date -Iseconds)"
  echo "repo_root=${REPO_ROOT}"
  echo "run_date=${RUN_DATE}"
  echo "python=$(command -v python3)"
  cd "${REPO_ROOT}"
  python3 -m quant_trade_system.nightly_quant_orders --date "${RUN_DATE}"
  echo "finished_at=$(TZ=Asia/Shanghai date -Iseconds)"
} >>"${LOG_FILE}" 2>&1

python3 -m quant_trade_system.nightly_health_check --date "${RUN_DATE}" --repo-root "${REPO_ROOT}" --json >>"${LOG_FILE}" 2>&1
