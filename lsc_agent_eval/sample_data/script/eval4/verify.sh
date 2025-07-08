#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${1:-openshift-lightspeed}"

echo "verify: checking namespace ${NAMESPACE}"
oc get ns "${NAMESPACE}" -o jsonpath='{.status.phase}{"\n"}'