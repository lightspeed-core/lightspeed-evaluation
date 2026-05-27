#!/usr/bin/env bash
set -euo pipefail

# Deploy agentic infrastructure + OOMKill test workload for integration tests.
# All operator-level resources use an "eval-" prefix to avoid conflicts with
# existing cluster resources.
#
# Required env vars:
#   GCP_CREDENTIALS_FILE          — path to GCP credentials JSON file
#                                   (default: ~/.config/gcloud/application_default_credentials.json)
#   ANTHROPIC_VERTEX_PROJECT_ID   — GCP project ID for Vertex AI
#   SANDBOX_IMAGE                 — sandbox container image URL
#
# Optional env vars:
#   CLOUD_ML_REGION  — default: global
#   AGENT_MODEL      — default: claude-opus-4-6

OPERATOR_NS="openshift-lightspeed"
TEST_NS="lightspeed-evaluation-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCP_CREDENTIALS_FILE="${GCP_CREDENTIALS_FILE:-$HOME/.config/gcloud/application_default_credentials.json}"
CLOUD_ML_REGION="${CLOUD_ML_REGION:-global}"
AGENT_MODEL="${AGENT_MODEL:-claude-opus-4-6}"

for var in ANTHROPIC_VERTEX_PROJECT_ID SANDBOX_IMAGE; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set" >&2
    exit 1
  fi
done

if [ ! -f "$GCP_CREDENTIALS_FILE" ]; then
  echo "ERROR: GCP credentials file not found: $GCP_CREDENTIALS_FILE" >&2
  exit 1
fi

# 1. Test namespace + OOMKill workload (static fixtures, already test-scoped)
oc apply -f "$SCRIPT_DIR/../fixtures/namespace.yaml"
oc apply -f "$SCRIPT_DIR/../fixtures/oomkill-demo.yaml"


# 2. Secret (GCP credentials) — prefixed name in operator namespace
oc create secret generic eval-llm-credentials \
  --from-file=credentials.json="$GCP_CREDENTIALS_FILE" \
  --from-literal=ANTHROPIC_VERTEX_PROJECT_ID="$ANTHROPIC_VERTEX_PROJECT_ID" \
  --from-literal=CLOUD_ML_REGION="$CLOUD_ML_REGION" \
  -n "$OPERATOR_NS" --dry-run=client -o yaml | oc apply -f -

# 3. LLMProvider — references prefixed secret
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
EOF

# 4. Agent — references prefixed LLMProvider
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

# 6. Wait for OOMKill workload to start crashing
echo "Waiting for oomkill-demo pod to appear..."
oc wait --for=condition=Available=false deployment/oomkill-demo \
  -n "$TEST_NS" --timeout=60s 2>/dev/null || true
sleep 10
echo "Setup complete."
