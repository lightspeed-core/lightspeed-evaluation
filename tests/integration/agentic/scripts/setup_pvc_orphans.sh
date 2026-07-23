#!/usr/bin/env bash
set -euo pipefail

# Deploy orphaned-PVCs analysis fixture (one claim in use, two unattached). Requires a default StorageClass.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_default_storageclass

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/pvc-orphans-demo.yaml"

echo "Waiting for pvc-user-app to become available (binds the active claim)..."
wait_for_deployment_available pvc-user-app 180
echo "Setup complete."
