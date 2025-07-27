#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${1:-openshift-lightspeed}"

echo "setup: ensuring namespace exists"
# `oc create ns` returns exit code 1 if the ns is present; ignore that specific case
oc get ns "${NAMESPACE}" >/dev/null 2>&1 || oc create ns "${NAMESPACE}"