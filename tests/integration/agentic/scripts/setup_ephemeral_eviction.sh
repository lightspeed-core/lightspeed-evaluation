#!/usr/bin/env bash
set -euo pipefail

# Deploy pod-eviction (emptyDir exceeds ephemeral-storage sizeLimit) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/ephemeral-eviction-demo.yaml"

echo "Waiting for ephemeral-eviction-demo pod to be evicted (kubelet sweep can take 1-2m)..."
wait_for_evicted ephemeral-eviction-demo 210
echo "Setup complete."
