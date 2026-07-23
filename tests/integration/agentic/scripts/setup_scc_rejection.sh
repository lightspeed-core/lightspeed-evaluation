#!/usr/bin/env bash
set -euo pipefail

# Deploy SCC-rejected test workload (pod template demands privileged root the restricted SCC forbids). Requires OpenShift SCC admission.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_api_group security.openshift.io securitycontextconstraints

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/scc-demo.yaml"

echo "Confirming pod creation for scc-demo is rejected by admission..."
wait_for_no_pods_created "scc-demo" 60
echo "Setup complete."
