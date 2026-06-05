#!/usr/bin/env bash
set -euo pipefail

# Tear down CrashLoopBackOff (liveness probe) test workload and Claude/Vertex AI infrastructure.

export OPERATOR_NS="openshift-lightspeed"
export TEST_NS="lightspeed-evaluation-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up CrashLoopBackOff probe integration test resources..."
oc delete deployment crashloop-probe-demo -n "$TEST_NS" --ignore-not-found

# shellcheck source=_cleanup_infra-claude-vertex.sh
source "$SCRIPT_DIR/_cleanup_infra-claude-vertex.sh"
echo "Cleanup complete."
