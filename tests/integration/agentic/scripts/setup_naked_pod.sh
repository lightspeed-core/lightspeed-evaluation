#!/usr/bin/env bash
set -euo pipefail

# Deploy "naked pod" (no probes, no resources, floating tag) hardening test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/naked-pod-demo.yaml"

wait_for_deployment_available "naked-pod-demo" 120
echo "Setup complete."
