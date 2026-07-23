#!/usr/bin/env bash
set -euo pipefail

# Deploy Pending pod (unsatisfiable nodeSelector) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/nodeselector-demo.yaml"

echo "Waiting for nodeselector-demo pod to be marked Unschedulable..."
wait_for_unschedulable "nodeselector-demo" 120
echo "Setup complete."
