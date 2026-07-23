#!/usr/bin/env bash
set -euo pipefail

# Deploy stuck rollout (bad image pushed over a healthy deployment, progress deadline exceeded) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/stuck-rollout-demo.yaml"

wait_for_deployment_available stuck-rollout-demo 120

echo "Rolling out a broken image to wedge the rollout..."
oc set image deployment/stuck-rollout-demo \
  nginx=docker.io/nginxinc/nginx-unprivileged:9.99-broken -n "$TEST_NS"

echo "Waiting for the rollout to exceed its progress deadline..."
oc wait --for=condition=Progressing=false deployment/stuck-rollout-demo \
  -n "$TEST_NS" --timeout=120s
echo "Setup complete."
