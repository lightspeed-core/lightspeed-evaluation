#!/usr/bin/env bash
set -euo pipefail

# Deploy the recurring maintenance-window fixture (batch-processor replays
# two days of history with a nightly 03:00-03:05 UTC failure window).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/batch-processor-demo.yaml"

wait_for_deployment_available batch-processor 120
echo "Waiting for the replayed history to be present in the logs..."
wait_for_log_match_full batch-processor 'ERROR Batch submission failed: connection timed out' 120
wait_for_log_match_full batch-processor 'INFO batch-processor resumed live processing' 120
echo "Setup complete."
