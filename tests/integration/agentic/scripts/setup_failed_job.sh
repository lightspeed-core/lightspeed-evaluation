#!/usr/bin/env bash
set -euo pipefail

# Deploy the failed-Job fixture: prod-db healthy on 8080, Service still
# advertising the decommissioned port 3333, Job pinned to 3333 so it
# exhausts its backoffLimit with connection-refused errors.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/failed-job-demo.yaml"

wait_for_deployment_available prod-db 120
wait_for_endpoints prod-db 120
echo "Waiting for inventory-sync-validator to exhaust its backoffLimit..."
wait_for_job_failed inventory-sync-validator 300
echo "Setup complete."
