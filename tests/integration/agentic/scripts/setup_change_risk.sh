#!/usr/bin/env bash
set -euo pipefail

# Deploy change-risk pre-flight fixture (healthy deployment + ConfigMap carrying a regressive proposed manifest).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/change-risk-demo.yaml"

wait_for_deployment_available "orders-api" 120
echo "Setup complete."
