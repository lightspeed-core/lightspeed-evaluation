#!/usr/bin/env bash
set -euo pipefail

# Deploy agentic infrastructure + OOMKill test workload for integration tests
# using OpenAI as the LLM provider.
# All operator-level resources use an "eval-" prefix to avoid conflicts with
# existing cluster resources.
#
# Required env vars:
#   OPENAI_API_KEY   — OpenAI API key
#   SANDBOX_IMAGE    — sandbox container image URL
#
# Optional env vars:
#   AGENT_MODEL      — default: gpt-4o
#
# To test with Claude via Vertex AI instead, use
# setup_subprocess_fixtures-claude-vertex.sh and set the corresponding env vars
# (GCP_CREDENTIALS_FILE, ANTHROPIC_VERTEX_PROJECT_ID, SANDBOX_IMAGE).

OPERATOR_NS="openshift-lightspeed"
TEST_NS="lightspeed-evaluation-test"
AGENT_MODEL="${AGENT_MODEL:-gpt-5.2}"

for var in OPENAI_API_KEY SANDBOX_IMAGE; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set" >&2
    exit 1
  fi
done

# 1. Test namespace + OOMKill workload (static fixtures, already test-scoped)
oc apply -f ../fixtures/namespace.yaml
oc apply -f ../fixtures/oomkill-demo.yaml

# 2. Secret (OpenAI API key + provider override) — prefixed name in operator namespace
# LIGHTSPEED_AGENT_PROVIDER and OPENAI_MODEL are injected via envFrom so the
# sandbox picks the OpenAI provider instead of defaulting to claude.
oc create secret generic eval-llm-credentials \
  --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
  --from-literal=LIGHTSPEED_AGENT_PROVIDER="openai" \
  --from-literal=OPENAI_MODEL="$AGENT_MODEL" \
  -n "$OPERATOR_NS" --dry-run=client -o yaml | oc apply -f -

# 3. LLMProvider — references prefixed secret
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

# 4. Agent — references prefixed LLMProvider
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
