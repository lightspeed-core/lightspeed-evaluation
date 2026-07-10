#!/usr/bin/env bash
set -euo pipefail

# Tear down broken alert expression test resources and provider infrastructure.
# OPERATOR_NS/TEST_NS may be overridden by the caller's environment.

TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Cleaning up broken_alert_expr integration test resources..."
if oc get crd prometheusrules.monitoring.coreos.com >/dev/null 2>&1; then
  oc delete prometheusrule silent-alert-rules -n "$TEST_NS" --ignore-not-found
fi

# shellcheck source=_cleanup_infra.sh
source "$SCRIPT_DIR/_cleanup_infra.sh"
echo "Cleanup complete."
