#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${1:-openshift-lightspeed}"

echo "teardown: deleting namespace ${NAMESPACE}"
oc delete ns "${NAMESPACE}" --ignore-not-found