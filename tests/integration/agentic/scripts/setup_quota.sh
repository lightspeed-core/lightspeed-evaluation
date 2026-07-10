#!/usr/bin/env bash
set -euo pipefail

# Deploy ResourceQuota exhaustion test resources (quota + blocker consuming it, then a victim that cannot create pods).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/quota-policy.yaml"

echo "Waiting for quota-blocker to consume the pod quota..."
wait_for_deployment_available quota-blocker 120

oc apply -f "$SCRIPT_DIR/../fixtures/quota-victim.yaml"

echo "Confirming quota-victim-app pod creation is rejected by the quota..."
wait_for_no_pods_created quota-victim-app 60
echo "Setup complete."
