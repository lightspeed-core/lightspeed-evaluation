#!/usr/bin/env bash
# Common infrastructure setup for OpenAI provider.
# Sourced by per-scenario setup scripts — do NOT run directly.
#
# Deploys: namespace, Secret, LLMProvider, Agent.
# All operator-level resources use an "eval-" prefix.
#
# Required env vars:
#   OPENAI_API_KEY   — OpenAI API key
#
# Optional env vars:
#   AGENT_MODEL      — default: gpt-5.2

set -euo pipefail

export OPERATOR_NS="openshift-lightspeed"
export TEST_NS="lightspeed-evaluation-test"
AGENT_MODEL="${AGENT_MODEL:-gpt-5.2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set" >&2
  exit 1
fi

# 1. Test namespace
oc apply -f "$SCRIPT_DIR/../fixtures/namespace.yaml"

# 2. Secret (OpenAI API key + provider override)
cat <<EOF | oc apply -n "$OPERATOR_NS" -f -
apiVersion: v1
kind: Secret
metadata:
  name: eval-llm-credentials
type: Opaque
stringData:
  OPENAI_API_KEY: ${OPENAI_API_KEY}
  LIGHTSPEED_AGENT_PROVIDER: openai
  OPENAI_MODEL: ${AGENT_MODEL}
EOF

# 3. LLMProvider
cat <<EOF | oc apply -f -
apiVersion: agentic.openshift.io/v1alpha1
kind: LLMProvider
metadata:
  name: eval-openai
spec:
  type: OpenAI
  openAI:
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
    name: eval-openai
  model: $AGENT_MODEL
  timeouts:
    analysisSeconds: 300
    executionSeconds: 600
    verificationSeconds: 300
  maxTurns: 200
EOF

echo "Infrastructure setup complete (OpenAI)."
