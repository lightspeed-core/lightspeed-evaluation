#!/usr/bin/env bash
# Common infrastructure setup for the direct Anthropic API provider.
# Sourced by per-scenario setup scripts — do NOT run directly.
#
# Deploys: namespace, Secret, LLMProvider, Agent.
# All operator-level resources use an "eval-" prefix.
#
# Required env vars:
#   ANTHROPIC_API_KEY — Anthropic API key
#
# Optional env vars:
#   AGENT_MODEL       — default: claude-sonnet-5

set -euo pipefail

export OPERATOR_NS="${OPERATOR_NS:-openshift-lightspeed}"
export TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
AGENT_MODEL="${AGENT_MODEL:-claude-sonnet-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set" >&2
  exit 1
fi

# 1. Test namespace
oc apply -f "$SCRIPT_DIR/../fixtures/namespace.yaml"

# 2. Secret (Anthropic API key)
cat <<EOF | oc apply -n "$OPERATOR_NS" -f -
apiVersion: v1
kind: Secret
metadata:
  name: eval-llm-credentials
type: Opaque
stringData:
  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
EOF

# 3. LLMProvider
cat <<EOF | oc apply -f -
apiVersion: agentic.openshift.io/v1alpha1
kind: LLMProvider
metadata:
  name: eval-anthropic
spec:
  type: Anthropic
  anthropic:
    credentialsSecret:
      name: eval-llm-credentials
EOF

# 4. Agent
cat <<EOF | oc apply -f -
apiVersion: agentic.openshift.io/v1alpha1
kind: Agent
metadata:
  name: eval-default
spec:
  llmProvider:
    name: eval-anthropic
  model: $AGENT_MODEL
  timeouts:
    analysisSeconds: 300
    executionSeconds: 600
    verificationSeconds: 300
  maxTurns: 200
EOF

echo "Infrastructure setup complete (Anthropic)."
