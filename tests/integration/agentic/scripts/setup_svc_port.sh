#!/usr/bin/env bash
set -euo pipefail

# Deploy service-with-wrong-targetPort (connections refused) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/svc-port-demo.yaml"

echo "Waiting for svc-port-demo deployment to become available..."
oc wait --for=condition=Available deployment/svc-port-demo \
  -n "$TEST_NS" --timeout=120s

echo "Waiting for svc-port-svc endpoints (selector is correct, port is not)..."
wait_for_endpoints svc-port-svc 60
echo "Setup complete."
