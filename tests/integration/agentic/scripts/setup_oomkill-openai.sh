#!/usr/bin/env bash
set -euo pipefail

# Deploy OOMKill test workload with OpenAI provider infrastructure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra-openai.sh
source "$SCRIPT_DIR/_setup_infra-openai.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/oomkill-demo.yaml"

echo "Waiting for oomkill-demo pod to reach OOMKilled or CrashLoopBackOff..."
TIMEOUT=120
INTERVAL=5
ELAPSED=0
while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
  TERMINATED_REASON=$(oc get pod -l app=oomkill-demo -n "$TEST_NS" \
    -o jsonpath='{.items[0].status.containerStatuses[0].state.terminated.reason}' 2>/dev/null || true)
  WAITING_REASON=$(oc get pod -l app=oomkill-demo -n "$TEST_NS" \
    -o jsonpath='{.items[0].status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)

  if [ "$TERMINATED_REASON" = "OOMKilled" ] || [ "$WAITING_REASON" = "CrashLoopBackOff" ]; then
    echo "Pod reached expected state (terminated=$TERMINATED_REASON, waiting=$WAITING_REASON)."
    echo "Setup complete."
    exit 0
  fi

  sleep "$INTERVAL"
  ELAPSED=$((ELAPSED + INTERVAL))
done

echo "ERROR: Timed out after ${TIMEOUT}s waiting for OOMKilled or CrashLoopBackOff." >&2
exit 1
