#!/usr/bin/env bash
set -euo pipefail

# Deploy mixed-hygiene namespace sweep fixture (three workloads with guardrail gaps, one compliant control).
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/hygiene-mixed-demo.yaml"

echo "Waiting for all hygiene-sweep deployments to become available..."
wait_for_deployments_available 150 web-frontend batch-worker legacy-api billing-api
echo "Setup complete."
