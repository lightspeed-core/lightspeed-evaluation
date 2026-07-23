#!/usr/bin/env bash
set -euo pipefail

# Deploy CrashLoopBackOff (liveness probe targeting the wrong port) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/probe-port-demo.yaml"

echo "Waiting for probe-port-demo to exhibit the fault..."
wait_for_container_state "probe-port-demo" 'CrashLoopBackOff' 180
echo "Setup complete."
