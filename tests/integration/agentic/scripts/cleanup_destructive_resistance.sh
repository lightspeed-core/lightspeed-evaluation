#!/usr/bin/env bash
set -euo pipefail

# Tear down destructive-suggestion resistance test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up destructive_resistance integration test resources..."
oc delete -f "$SCRIPT_DIR/../fixtures/fragile-app-demo.yaml" --ignore-not-found
# --timeout so a finalizer-stuck claim cannot wedge cleanup past the 300s
# kill before infra teardown runs.
oc delete pvc --all -n "$TEST_NS" --timeout=120s || true
rm -f "${TMPDIR:-/tmp}/lightspeed-eval-state/fragile-data.uid"

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
