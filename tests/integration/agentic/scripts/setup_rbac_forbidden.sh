#!/usr/bin/env bash
set -euo pipefail

# Deploy API-403 test workload (ServiceAccount without RBAC for the pod list its app performs).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/rbac-forbidden-demo.yaml"

wait_for_deployment_available rbac-forbidden-demo 120

echo "Waiting for the app to log 403 responses..."
wait_for_log_match rbac-forbidden-demo 'HTTP 403' 60
echo "Setup complete."
