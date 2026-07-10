#!/usr/bin/env bash
set -euo pipefail

# Deploy broken HPA (deployment without resource requests, metrics unavailable) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/hpa-demo.yaml"

wait_for_deployment_available hpa-underspecified-demo 120

echo "Waiting for the HPA to report failed metrics (ScalingActive=False)..."
oc wait --for=condition=ScalingActive=false hpa/hpa-underspecified-demo \
  -n "$TEST_NS" --timeout=120s
echo "Setup complete."
