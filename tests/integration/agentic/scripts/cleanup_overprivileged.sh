#!/usr/bin/env bash
set -euo pipefail

# Tear down overprivileged binding test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up overprivileged integration test resources..."
oc delete -f "$SCRIPT_DIR/../fixtures/overprivileged-demo.yaml" --ignore-not-found

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
