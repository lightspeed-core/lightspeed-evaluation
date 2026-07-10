#!/usr/bin/env bash
# Common infrastructure setup for Claude via Vertex AI.
# Sourced by per-scenario setup scripts — do NOT run directly.
#
# Deploys: namespace, Secret, LLMProvider, Agent.
# All operator-level resources use an "eval-" prefix.
#
# Required env vars:
#   GCP_CREDENTIALS_FILE          — path to GCP credentials JSON file
#                                   (default: ~/.config/gcloud/application_default_credentials.json)
#   ANTHROPIC_VERTEX_PROJECT_ID   — GCP project ID for Vertex AI
#
# Optional env vars:
#   CLOUD_ML_REGION  — default: global
#   AGENT_MODEL      — default: claude-opus-4-6

set -euo pipefail

export OPERATOR_NS="${OPERATOR_NS:-openshift-lightspeed}"
export TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCP_CREDENTIALS_FILE="${GCP_CREDENTIALS_FILE:-$HOME/.config/gcloud/application_default_credentials.json}"
CLOUD_ML_REGION="${CLOUD_ML_REGION:-global}"
AGENT_MODEL="${AGENT_MODEL:-claude-opus-4-6}"

if [ -z "${ANTHROPIC_VERTEX_PROJECT_ID:-}" ]; then
  echo "ERROR: ANTHROPIC_VERTEX_PROJECT_ID is not set" >&2
  exit 1
fi

if [ ! -f "$GCP_CREDENTIALS_FILE" ]; then
  echo "ERROR: GCP credentials file not found: $GCP_CREDENTIALS_FILE" >&2
  exit 1
fi

# 1. Test namespace
oc apply -f "$SCRIPT_DIR/../fixtures/namespace.yaml"

# 2. Secret (GCP credentials)
oc create secret generic eval-llm-credentials \
  --from-file=GOOGLE_APPLICATION_CREDENTIALS="$GCP_CREDENTIALS_FILE" \
  --from-literal=ANTHROPIC_VERTEX_PROJECT_ID="$ANTHROPIC_VERTEX_PROJECT_ID" \
  --from-literal=CLOUD_ML_REGION="$CLOUD_ML_REGION" \
  -n "$OPERATOR_NS" --dry-run=client -o yaml | oc apply -f -

# 3. LLMProvider
cat <<EOF | oc apply -f -
apiVersion: agentic.openshift.io/v1alpha1
kind: LLMProvider
metadata:
  name: eval-vertex-ai
spec:
  type: GoogleCloudVertex
  googleCloudVertex:
    credentialsSecret:
      name: eval-llm-credentials
    projectID: $ANTHROPIC_VERTEX_PROJECT_ID
    region: $CLOUD_ML_REGION
    modelProvider: Anthropic
EOF

# 4. Agent
cat <<EOF | oc apply -f -
apiVersion: agentic.openshift.io/v1alpha1
kind: Agent
metadata:
  name: eval-default
spec:
  llmProvider:
    name: eval-vertex-ai
  model: $AGENT_MODEL
  timeouts:
    analysisSeconds: 600
    executionSeconds: 600
    verificationSeconds: 600
  maxTurns: 200
EOF

echo "Infrastructure setup complete (Claude/Vertex AI)."
