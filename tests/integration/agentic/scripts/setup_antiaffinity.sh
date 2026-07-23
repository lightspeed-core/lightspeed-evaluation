#!/usr/bin/env bash
set -euo pipefail

# Deploy Pending pods (required anti-affinity exceeding node count) test workload. Replicas are scaled to node count + 3 so the fault exists on any cluster size; requires >= 3 schedulable nodes for the graded remediation (reduce to 3 spread replicas).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/antiaffinity-demo.yaml"

NODE_COUNT=$(oc get nodes --no-headers | wc -l | tr -d ' ')
REPLICAS=$((NODE_COUNT + 3))
echo "Cluster has ${NODE_COUNT} nodes; scaling antiaffinity-demo to ${REPLICAS} replicas..."
oc scale deployment/antiaffinity-demo -n "$TEST_NS" --replicas="$REPLICAS"

echo "Waiting for surplus antiaffinity-demo pods to be Unschedulable..."
wait_for_unschedulable antiaffinity-demo 120

# The remediation (3 replicas, one per node) is only achievable if at
# least 3 pods actually scheduled — an empirical >=3-schedulable-nodes
# check, since anti-affinity permits one pod per node.
SCHEDULED=$(oc get pods -l app=antiaffinity-demo -n "$TEST_NS" \
  -o jsonpath='{range .items[*].status.conditions[?(@.type=="PodScheduled")]}{.status}{"\n"}{end}' \
  | grep -c True || true)
if [ "$SCHEDULED" -lt 3 ]; then
  echo "ERROR: only ${SCHEDULED} pods scheduled — this scenario needs >= 3 schedulable nodes" >&2
  exit 1
fi
echo "Setup complete."
