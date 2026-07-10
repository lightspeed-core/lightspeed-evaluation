#!/usr/bin/env bash
set -euo pipefail

# Deploy Route-to-wrong-port (router 503) test workload. Requires the OpenShift Route API.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_api_group route.openshift.io routes

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/route-port-demo.yaml"

wait_for_deployment_available route-port-demo 120
wait_for_route_admitted route-port-demo 60
echo "Setup complete."
