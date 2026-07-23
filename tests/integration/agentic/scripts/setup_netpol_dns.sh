#!/usr/bin/env bash
set -euo pipefail

# Deploy default-deny-egress-without-DNS test workload (app cannot resolve hostnames).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/netpol-dns-demo.yaml"

wait_for_deployment_available netpol-dns-demo 120

echo "Waiting for the app to log DNS resolution failures..."
wait_for_log_match netpol-dns-demo 'DNS resolution failed' 60
echo "Setup complete."
