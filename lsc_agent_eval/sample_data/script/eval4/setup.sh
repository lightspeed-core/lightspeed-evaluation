#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${1:-openshift-lightspeed}"

echo "setup: ensuring clean slate for ${NAMESPACE}"
oc delete ns "${NAMESPACE}" --ignore-not-found