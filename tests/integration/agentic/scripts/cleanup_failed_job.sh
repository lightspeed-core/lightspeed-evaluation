#!/usr/bin/env bash
set -euo pipefail

# Tear down failed-Job test resources (including any corrected Job the agent
# created during remediation) and provider infrastructure. OPERATOR_NS/
# TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up failed_job integration test resources..."
oc delete job inventory-sync-validator -n "$TEST_NS" --ignore-not-found
oc delete jobs -l app=inventory-sync-validator -n "$TEST_NS" --ignore-not-found
oc delete service prod-db -n "$TEST_NS" --ignore-not-found
oc delete deployment prod-db -n "$TEST_NS" --ignore-not-found

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
