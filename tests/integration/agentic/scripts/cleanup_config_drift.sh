#!/usr/bin/env bash
set -euo pipefail

# Tear down config-drift test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up config_drift integration test resources..."
oc delete deployment gateway-proxy -n "$TEST_NS" --ignore-not-found
oc delete configmap gateway-proxy-config -n "$TEST_NS" --ignore-not-found

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
