#!/usr/bin/env bash
set -euo pipefail

# Deploy red-herring test workloads (real crashloop plus an intentionally-not-Ready canary decoy).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/red-herring-demo.yaml"

echo "Waiting for herring-app to exhibit its crash fault..."
wait_for_container_state herring-app 'CrashLoopBackOff|Error' 120

echo "Waiting for the canary decoy to be Running but not Ready..."
wait_for_running_not_ready maintenance-canary 120
echo "Setup complete."
