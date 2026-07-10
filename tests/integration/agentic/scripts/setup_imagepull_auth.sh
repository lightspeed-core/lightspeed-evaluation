#!/usr/bin/env bash
set -euo pipefail

# Deploy ImagePullBackOff (unauthorized private registry, missing pull secret) test workload.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/imagepull-auth-demo.yaml"

echo "Waiting for imagepull-auth-demo to exhibit the fault..."
wait_for_container_state "imagepull-auth-demo" 'ErrImagePull|ImagePullBackOff' 120
echo "Setup complete."
