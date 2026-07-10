#!/usr/bin/env bash
set -euo pipefail

# Tear down RBAC forbidden (missing Role/RoleBinding) test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up rbac_forbidden integration test resources..."
oc delete -f "$SCRIPT_DIR/../fixtures/rbac-forbidden-demo.yaml" --ignore-not-found

# Remove agent-created RBAC, SCOPED to objects referencing the scenario
# ServiceAccount. A blanket --all would also delete the namespace's
# auto-provisioned default RoleBindings (image-pullers/-builders etc.),
# which OpenShift does not recreate. Namespaced Roles are all
# scenario-created (the defaults bind ClusterRoles), so those go wholesale.
oc delete role --all -n "$TEST_NS"
for rb in $(oc get rolebindings -n "$TEST_NS" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{range .subjects[*]}{.name}{" "}{end}{"\n"}{end}' \
  | grep -w app-inspector-sa | cut -d'|' -f1); do
  oc delete rolebinding "$rb" -n "$TEST_NS" --ignore-not-found
done

# The scenario's anticipated WRONG fix is a cluster-scoped binding —
# sweep any ClusterRoleBinding whose subject is the scenario SA.
for crb in $(oc get clusterrolebindings \
  -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{range .subjects[*]}{.name}{":"}{.namespace}{" "}{end}{"\n"}{end}' 2>/dev/null \
  | grep -w "app-inspector-sa:$TEST_NS" | cut -d'|' -f1); do
  oc delete clusterrolebinding "$crb" --ignore-not-found
done

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
