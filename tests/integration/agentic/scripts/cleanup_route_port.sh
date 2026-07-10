#!/usr/bin/env bash
set -euo pipefail

# Tear down Route port mismatch test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up route_port integration test resources..."
if oc api-resources --api-group=route.openshift.io 2>/dev/null | grep -q routes; then
  oc delete -f "$SCRIPT_DIR/../fixtures/route-port-demo.yaml" --ignore-not-found
fi

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
