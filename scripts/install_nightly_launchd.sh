#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LABEL="${QTS_LAUNCHD_LABEL:-com.quant-trading-system.nightly-orders}"
RUNTIME_ROOT="${QTS_LAUNCHD_RUNTIME_ROOT:-${HOME}/Library/Application Support/quant-trading-system/runtime}"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
RUN_SCRIPT="${RUNTIME_ROOT}/scripts/run_nightly_quant_orders.sh"
NIGHTLY_PATH="${QTS_NIGHTLY_PATH:-/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
export QTS_NIGHTLY_PATH="${NIGHTLY_PATH}"
PYTHON_BIN="$(bash "${REPO_ROOT}/scripts/resolve_nightly_python.sh")"

mkdir -p "${HOME}/Library/LaunchAgents" "${RUNTIME_ROOT}" "${RUNTIME_ROOT}/state/nightly_logs"
rsync -a --delete \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "state/nightly_logs" \
  --exclude "state/nightly_reports" \
  --exclude "state/nightly_status" \
  "${REPO_ROOT}/" "${RUNTIME_ROOT}/"
chmod +x "${RUN_SCRIPT}"

cat >"${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RUN_SCRIPT}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${RUNTIME_ROOT}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>20</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <key>StartInterval</key>
  <integer>3600</integer>
  <key>StandardOutPath</key>
  <string>${RUNTIME_ROOT}/state/nightly_logs/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${RUNTIME_ROOT}/state/nightly_logs/launchd_stderr.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${NIGHTLY_PATH}</string>
    <key>PYTHONPATH</key>
    <string>${RUNTIME_ROOT}</string>
    <key>QTS_PYTHON_BIN</key>
    <string>${PYTHON_BIN}</string>
    <key>QTS_FUTURES_SETTLE_FALLBACK</key>
    <string>daily_main_contract</string>
    <key>QTS_NIGHTLY_CATCH_UP</key>
    <string>1</string>
    <key>QTS_NIGHTLY_PREVENT_SLEEP</key>
    <string>1</string>
    <key>QTS_NIGHTLY_SKIP_EXISTING</key>
    <string>1</string>
    <key>TZ</key>
    <string>Asia/Shanghai</string>
  </dict>
</dict>
</plist>
PLIST

LAUNCHD_DOMAIN="gui/$(id -u)"
launchctl bootout "${LAUNCHD_DOMAIN}" "${PLIST_PATH}" >/dev/null 2>&1 || launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl bootstrap "${LAUNCHD_DOMAIN}" "${PLIST_PATH}" >/dev/null 2>&1 || launchctl load "${PLIST_PATH}"
launchctl enable "${LAUNCHD_DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
launchctl print "${LAUNCHD_DOMAIN}/${LABEL}" | sed -n '1,80p' || true

echo "Installed ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Run script: ${RUN_SCRIPT}"
echo "Runtime: ${RUNTIME_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Logs: ${RUNTIME_ROOT}/state/nightly_logs"
