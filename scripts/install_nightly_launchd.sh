#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LABEL="${QTS_LAUNCHD_LABEL:-com.quant-trading-system.nightly-orders}"
RUNTIME_ROOT="${QTS_LAUNCHD_RUNTIME_ROOT:-${HOME}/Library/Application Support/quant-trading-system/runtime}"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
RUN_SCRIPT="${RUNTIME_ROOT}/scripts/run_nightly_quant_orders.sh"

mkdir -p "${HOME}/Library/LaunchAgents" "${RUNTIME_ROOT}" "${RUNTIME_ROOT}/state/nightly_logs"
rsync -a --delete \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "state/nightly_logs" \
  --exclude "state/nightly_reports" \
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
  <key>StandardOutPath</key>
  <string>${RUNTIME_ROOT}/state/nightly_logs/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${RUNTIME_ROOT}/state/nightly_logs/launchd_stderr.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>${RUNTIME_ROOT}</string>
    <key>QTS_FUTURES_SETTLE_FALLBACK</key>
    <string>daily_main_contract</string>
    <key>TZ</key>
    <string>Asia/Shanghai</string>
  </dict>
</dict>
</plist>
PLIST

launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl load "${PLIST_PATH}"
launchctl list | grep "${LABEL}" || true

echo "Installed ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Run script: ${RUN_SCRIPT}"
echo "Runtime: ${RUNTIME_ROOT}"
echo "Logs: ${RUNTIME_ROOT}/state/nightly_logs"
