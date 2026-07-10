#!/usr/bin/env bash
set -euo pipefail

# Deploy Init:CrashLoopBackOff (failing init container) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/init-container-demo.yaml"

echo "Waiting for init-container-demo init container to fail..."
wait_for_init_container_state init-container-demo 'CrashLoopBackOff|Error' 120
echo "Setup complete."
