#!/usr/bin/env bash
set -euo pipefail

# Deploy alert-coverage-gap analysis fixture (workload with a recording-rule-only PrometheusRule). Requires the PrometheusRule CRD.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_crd prometheusrules.monitoring.coreos.com

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/missing-alerts-demo.yaml"

wait_for_deployment_available payments-api 120
echo "Setup complete."
