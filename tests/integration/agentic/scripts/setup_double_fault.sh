#!/usr/bin/env bash
set -euo pipefail

# Deploy double-fault test workload (missing ConfigMap AND wrong liveness-probe port in one pod).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/double-fault-demo.yaml"

echo "Waiting for double-fault-demo to exhibit the fault..."
wait_for_container_state "double-fault-demo" 'CreateContainerConfigError' 120
echo "Setup complete."
