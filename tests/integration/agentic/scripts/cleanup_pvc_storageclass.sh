#!/usr/bin/env bash
set -euo pipefail

# Tear down Pending PVC (bad StorageClass) test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up pvc_storageclass integration test resources..."
oc delete -f "$SCRIPT_DIR/../fixtures/pvc-storageclass-demo.yaml" --ignore-not-found
# Also remove any replacement PVC the agent created under a new name,
# so it cannot pollute the later pvc_orphans classification scenario.
oc delete pvc --all -n "$TEST_NS" --timeout=120s || true

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
