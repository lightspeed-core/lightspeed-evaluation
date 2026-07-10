#!/usr/bin/env bash
set -euo pipefail

# Deploy Pending PVC (nonexistent StorageClass) test workload. Requires a default StorageClass for the remediation.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_default_storageclass

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/pvc-storageclass-demo.yaml"

echo "Waiting for pvc-app-demo pod to be unschedulable on the unbound claim..."
wait_for_unschedulable pvc-app-demo 120
echo "Setup complete."
