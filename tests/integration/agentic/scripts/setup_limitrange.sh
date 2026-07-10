#!/usr/bin/env bash
set -euo pipefail

# Deploy LimitRange conflict test resources (policy first so enforcement is active before the app).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/limitrange-policy.yaml"
sleep 5
oc apply -f "$SCRIPT_DIR/../fixtures/limitrange-app.yaml"

echo "Confirming limitrange-demo pod creation is rejected by the LimitRange..."
wait_for_no_pods_created limitrange-demo 60
echo "Setup complete."
