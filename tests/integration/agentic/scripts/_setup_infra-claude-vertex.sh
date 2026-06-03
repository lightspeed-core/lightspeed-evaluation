#!/usr/bin/env bash
# Common infrastructure setup for Claude via Vertex AI.
# Sourced by per-scenario setup scripts — do NOT run directly.
#
# Deploys: namespace, Secret, LLMProvider, Agent, SandboxTemplate.
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
#   SANDBOX_IMAGE    — sandbox container image URL (has default)

set -euo pipefail

export OPERATOR_NS="openshift-lightspeed"
export TEST_NS="lightspeed-evaluation-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCP_CREDENTIALS_FILE="${GCP_CREDENTIALS_FILE:-$HOME/.config/gcloud/application_default_credentials.json}"
CLOUD_ML_REGION="${CLOUD_ML_REGION:-global}"
AGENT_MODEL="${AGENT_MODEL:-claude-opus-4-6}"
# TODO: replace with a stable tag once a versioned release is available
SANDBOX_IMAGE="${SANDBOX_IMAGE:-quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/lightspeed-agentic-sandbox:d84b7970dc65ab3e66d52f3f2feeb1b3ec5b72eb}"

for var in ANTHROPIC_VERTEX_PROJECT_ID; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set" >&2
    exit 1
  fi
done

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
    analysisSeconds: 300
    executionSeconds: 600
    verificationSeconds: 300
  maxTurns: 200
EOF

# 5. SandboxTemplate
cat <<EOF | oc apply -f -
apiVersion: extensions.agents.x-k8s.io/v1alpha1
kind: SandboxTemplate
metadata:
  name: eval-lightspeed-agent
  namespace: $OPERATOR_NS
spec:
  networkPolicyManagement: Unmanaged
  podTemplate:
    spec:
      serviceAccountName: lightspeed-agent
      automountServiceAccountToken: true
      containers:
      - name: agent
        image: $SANDBOX_IMAGE
        imagePullPolicy: Always
        ports:
          - containerPort: 8080
            protocol: TCP
        env:
          - name: LIGHTSPEED_SKILLS_DIR
            value: /app/skills
        volumeMounts:
          - name: home
            mountPath: /home/agent
          - name: tmp
            mountPath: /tmp
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: "4"
            memory: 4Gi
      volumes:
      - name: home
        emptyDir: {}
      - name: tmp
        emptyDir: {}
EOF

echo "Infrastructure setup complete (Claude/Vertex AI)."
