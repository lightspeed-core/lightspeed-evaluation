#!/usr/bin/env bash
set -euo pipefail

# Tear down missing PVC reference test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up missing_pvc integration test resources..."
oc delete deployment missing-pvc-demo -n "$TEST_NS" --ignore-not-found
# Remove the agent-created claim (any name it chose).
oc delete pvc --all -n "$TEST_NS" --timeout=120s || true

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
