#!/usr/bin/env bash
set -euo pipefail

# Tear down integration test resources. Only deletes "eval-" prefixed operator
# resources to avoid touching anything that isn't ours.

OPERATOR_NS="openshift-lightspeed"
TEST_NS="lightspeed-evaluation-test"

echo "Cleaning up integration test resources..."

# Delete test workload + any leftover Proposals in test namespace
oc delete deployment oomkill-demo -n "$TEST_NS" --ignore-not-found
oc delete proposals --all -n "$TEST_NS" --ignore-not-found
oc delete proposalapprovals --all -n "$TEST_NS" --ignore-not-found

# Delete prefixed operator resources (reverse order of creation)
oc delete sandboxtemplate eval-lightspeed-agent -n "$OPERATOR_NS" --ignore-not-found
oc delete agent eval-default --ignore-not-found
oc delete llmprovider eval-vertex-ai --ignore-not-found
oc delete secret eval-llm-credentials -n "$OPERATOR_NS" --ignore-not-found

# Delete test namespace (cascades remaining namespaced resources)
oc delete namespace "$TEST_NS" --ignore-not-found

echo "Cleanup complete."
