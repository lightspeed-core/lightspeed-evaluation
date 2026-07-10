#!/usr/bin/env bash
# Provider-agnostic infrastructure cleanup.
# Sourced by per-scenario cleanup scripts — do NOT run directly.
#
# Deletes "eval-" prefixed operator resources for ALL providers so cleanup
# works regardless of which EVAL_PROVIDER the setup ran with.

OPERATOR_NS="${OPERATOR_NS:-openshift-lightspeed}"
TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"

oc delete agent eval-default --ignore-not-found
oc delete llmprovider eval-openai eval-anthropic eval-vertex-ai --ignore-not-found
oc delete secret eval-llm-credentials -n "$OPERATOR_NS" --ignore-not-found

# Optional full reset: also remove the test namespace so no namespace-scoped
# state (quotas, LimitRanges, secrets) leaks into the next run.
if [ "${EVAL_DELETE_NAMESPACE:-0}" = "1" ]; then
  oc delete namespace "$TEST_NS" --ignore-not-found
fi
