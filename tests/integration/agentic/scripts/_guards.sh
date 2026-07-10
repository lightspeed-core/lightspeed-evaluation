#!/usr/bin/env bash
# Cluster-capability guards for scenario setup scripts — sourced, do NOT run
# directly. Each guard exits non-zero with a clear message so scenarios fail
# fast BEFORE provider infrastructure (Secret/LLMProvider/Agent) is deployed.
# Source this before _setup_infra.sh.

require_crd() { # crd_name
  if ! oc get crd "$1" >/dev/null 2>&1; then
    echo "ERROR: required CRD $1 not found on this cluster" >&2
    exit 1
  fi
}

require_api_group() { # api_group resource
  if ! oc api-resources --api-group="$1" 2>/dev/null | grep -qw "$2"; then
    echo "ERROR: required API $2.$1 not available on this cluster" >&2
    exit 1
  fi
}

require_default_storageclass() {
  if ! oc get storageclass -o jsonpath='{.items[*].metadata.annotations.storageclass\.kubernetes\.io/is-default-class}' 2>/dev/null \
    | grep -q true; then
    echo "ERROR: no default StorageClass on this cluster" >&2
    exit 1
  fi
}
