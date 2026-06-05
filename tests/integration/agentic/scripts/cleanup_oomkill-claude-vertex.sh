#!/usr/bin/env bash
set -euo pipefail

# Tear down OOMKill test workload and Claude/Vertex AI provider infrastructure.

export OPERATOR_NS="openshift-lightspeed"
export TEST_NS="lightspeed-evaluation-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up OOMKill integration test resources..."
oc delete deployment oomkill-demo -n "$TEST_NS" --ignore-not-found

# shellcheck source=_cleanup_infra-claude-vertex.sh
source "$SCRIPT_DIR/_cleanup_infra-claude-vertex.sh"
echo "Cleanup complete."
