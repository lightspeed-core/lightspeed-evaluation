#!/usr/bin/env bash
set -euo pipefail

# Deploy service-with-no-endpoints (selector/label mismatch) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/svc-selector-demo.yaml"

echo "Waiting for svc-selector-demo deployment to become available..."
oc wait --for=condition=Available deployment/svc-selector-demo \
  -n "$TEST_NS" --timeout=120s

confirm_no_endpoints svc-selector-svc
echo "Setup complete."
