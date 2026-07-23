#!/usr/bin/env bash
set -euo pipefail

# Deploy NetworkPolicy-blocked in-namespace traffic test workloads (backend locked down, client without the allowed label).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/netpol-ingress-demo.yaml"

echo "Waiting for backend and client deployments to become available..."
# Shared deadline (a plain multi-resource oc wait applies --timeout per
# resource) keeps worst case under the 300s subprocess timeout.
wait_for_deployments_available 120 netpol-backend netpol-client

echo "Waiting for the client to log blocked requests..."
wait_for_log_match netpol-client 'backend unreachable' 60
echo "Setup complete."
