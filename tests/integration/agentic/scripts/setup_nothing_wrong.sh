#!/usr/bin/env bash
set -euo pipefail

# Deploy fully healthy workload for the false-positive scenario (request claims it is broken; correct outcome is no change).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/healthy-app-demo.yaml"

wait_for_deployment_available "healthy-app" 120
echo "Setup complete."
