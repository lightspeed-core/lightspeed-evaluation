#!/usr/bin/env bash
set -euo pipefail

# Tear down NetworkPolicy ingress lockdown test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up netpol_ingress integration test resources..."
oc delete -f "$SCRIPT_DIR/../fixtures/netpol-ingress-demo.yaml" --ignore-not-found
# Also remove any NetworkPolicies the agent created during remediation.
oc delete networkpolicy --all -n "$TEST_NS"

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
