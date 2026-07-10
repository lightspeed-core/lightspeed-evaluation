#!/usr/bin/env bash
set -euo pipefail

# Deploy the cross-namespace connection-pool exhaustion fixture: symptom
# (payments-api 503s) in TEST_NS, root cause (reporting-service leaking
# shared-db connections) plus an unrelated crashloop decoy in SHARED_NS.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

SHARED_NS="${SHARED_NS:-lightspeed-evaluation-shared}"

oc apply -f "$SCRIPT_DIR/../fixtures/conn-pool-demo.yaml"

echo "Waiting for shared-namespace workloads to start..."
TEST_NS="$SHARED_NS" wait_for_deployments_available 180 shared-db reporting-service
wait_for_deployment_available payments-api 120

# Roll reporting-service to the leaky release so `oc rollout history` shows
# the offending change (revision 2, v1.0.2).
oc set env "deployment/reporting-service" -n "$SHARED_NS" RELEASE=v1.0.2
oc annotate "deployment/reporting-service" -n "$SHARED_NS" \
  kubernetes.io/change-cause="deploy reporting-service v1.0.2" --overwrite

echo "Waiting for the fault chain to be observable..."
TEST_NS="$SHARED_NS" wait_for_log_match reporting-service 'held: [0-9]+, released: 0' 180
TEST_NS="$SHARED_NS" wait_for_log_match shared-db 'FATAL too many connections' 240
TEST_NS="$SHARED_NS" wait_for_container_state reconciliation-service 'CrashLoopBackOff|Error' 180
wait_for_log_match payments-api 'HTTP 503' 120
echo "Setup complete."
