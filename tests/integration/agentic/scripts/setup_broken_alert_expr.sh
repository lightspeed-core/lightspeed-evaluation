#!/usr/bin/env bash
set -euo pipefail

# Deploy silent-alert analysis fixture (alert expressions referencing misspelled metric names). Requires the PrometheusRule CRD.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_guards.sh
source "$SCRIPT_DIR/_guards.sh"
require_crd prometheusrules.monitoring.coreos.com

# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/broken-alert-rules.yaml"
echo "Setup complete."
