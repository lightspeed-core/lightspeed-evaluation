#!/usr/bin/env bash
set -euo pipefail

# Deploy right-sizing analysis fixture (idle nginx requesting 1 CPU / 1Gi per replica). Requires the pod metrics API (metrics-server).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_api_group metrics.k8s.io pods

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/rightsizing-demo.yaml"

wait_for_deployment_available oversized-api 120

echo "Waiting for metrics-server to report usage for oversized-api..."
wait_for_pod_metrics oversized-api 120
echo "Setup complete."
