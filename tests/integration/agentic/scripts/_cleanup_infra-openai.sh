#!/usr/bin/env bash
# Common infrastructure cleanup for OpenAI provider.
# Sourced by per-scenario cleanup scripts — do NOT run directly.
#
# Deletes "eval-" prefixed operator resources (reverse order of creation).

OPERATOR_NS="${OPERATOR_NS:-openshift-lightspeed}"
TEST_NS="${TEST_NS:-lightspeed-evaluation-test}"

oc delete proposals --all -n "$TEST_NS" --ignore-not-found
oc delete proposalapprovals --all -n "$TEST_NS" --ignore-not-found
oc delete agent eval-default --ignore-not-found
oc delete llmprovider eval-openai --ignore-not-found
oc delete secret eval-llm-credentials -n "$OPERATOR_NS" --ignore-not-found
