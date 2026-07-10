#!/usr/bin/env bash
set -euo pipefail

# Deploy the config-drift fixture (gateway-proxy hot-reloaded staging
# upstream hosts into production; pod stays Running and Ready).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/gateway-proxy-demo.yaml"

wait_for_deployment_available gateway-proxy 120
echo "Waiting for the hot-reload timeline and refused connections..."
wait_for_log_match_full gateway-proxy 'config change detected in /config/app.yaml' 120
wait_for_log_match gateway-proxy 'ECONNREFUSED' 120
echo "Setup complete."
