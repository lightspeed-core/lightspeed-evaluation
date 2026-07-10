#!/usr/bin/env bash
set -euo pipefail

# Tear down connection-pool exhaustion test resources (both namespaces) and
# provider infrastructure. OPERATOR_NS/TEST_NS may be overridden by the
# caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SHARED_NS="${SHARED_NS:-lightspeed-evaluation-shared}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up conn_pool integration test resources..."
oc delete deployment payments-api -n "$TEST_NS" --ignore-not-found
oc delete namespace "$SHARED_NS" --ignore-not-found

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
