#!/usr/bin/env bash
set -euo pipefail

# Deploy OOMKill test workload with Claude/Vertex AI provider infrastructure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra-claude-vertex.sh
source "$SCRIPT_DIR/_setup_infra-claude-vertex.sh"

oc apply -f "$SCRIPT_DIR/../fixtures/oomkill-demo.yaml"

echo "Waiting for oomkill-demo pod to appear..."
oc wait --for=condition=Available=false deployment/oomkill-demo \
  -n "$TEST_NS" --timeout=60s 2>/dev/null || true
sleep 10
echo "Setup complete."
