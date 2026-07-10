#!/usr/bin/env bash
set -euo pipefail

# Deploy FailedCreate (nonexistent PriorityClass reference) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/priorityclass-demo.yaml"

echo "Confirming pod creation for priorityclass-demo is rejected by admission..."
wait_for_no_pods_created "priorityclass-demo" 60
echo "Setup complete."
