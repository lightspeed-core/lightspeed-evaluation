#!/usr/bin/env bash
set -euo pipefail

# Deploy missing-PVC-reference test workload. Requires a default StorageClass for the remediation.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_default_storageclass

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/missing-pvc-demo.yaml"

echo "Waiting for missing-pvc-demo pod to be unschedulable on the missing claim..."
wait_for_unschedulable missing-pvc-demo 120
echo "Setup complete."
