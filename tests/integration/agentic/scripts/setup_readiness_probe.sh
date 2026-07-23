#!/usr/bin/env bash
set -euo pipefail

# Deploy never-Ready pod (broken readiness probe, service with no endpoints) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/readiness-probe-demo.yaml"

echo "Waiting for readiness-probe-demo pod to be Running but not Ready..."
wait_for_running_not_ready readiness-probe-demo 120
echo "Setup complete."
