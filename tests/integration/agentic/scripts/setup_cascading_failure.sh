#!/usr/bin/env bash
set -euo pipefail

# Deploy cascading-failure test workloads (frontend not Ready because the backend image cannot pull).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/cascade-demo.yaml"

echo "Waiting for the backend to exhibit its image-pull fault..."
wait_for_container_state cascade-backend 'ErrImagePull|ImagePullBackOff' 120

echo "Waiting for the frontend to be Running but not Ready..."
wait_for_running_not_ready cascade-frontend 120
echo "Setup complete."
