#!/usr/bin/env bash
set -euo pipefail

# Deploy needle-in-haystack sweep fixture (five workloads, exactly two broken).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/haystack-demo.yaml"

# Fault polls run FIRST: both faults surface within ~40s of the apply and
# overlap the healthy workloads' pull time, keeping the script's worst
# case (~60+60+120+applies) under the 300s subprocess timeout.
echo "Waiting for the two broken workloads to exhibit their faults..."
wait_for_container_state svc-charlie 'CreateContainerConfigError' 60
wait_for_container_state svc-echo 'ErrImagePull|ImagePullBackOff' 60

echo "Waiting for the three healthy workloads to become available..."
wait_for_deployments_available 120 svc-alpha svc-bravo svc-delta
echo "Setup complete."
