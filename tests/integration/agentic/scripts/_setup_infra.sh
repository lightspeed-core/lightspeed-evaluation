#!/usr/bin/env bash
# Provider dispatcher for infrastructure setup.
# Sourced by per-scenario setup scripts — do NOT run directly.
#
# Selects the provider infrastructure via the EVAL_PROVIDER env var
# (openai | anthropic | claude-vertex; default: openai) and sources the
# matching _setup_infra-<provider>.sh. Per-scenario scripts stay
# provider-agnostic; pick the provider at runtime:
#
#   EVAL_PROVIDER=anthropic uv run lightspeed-eval ...

EVAL_PROVIDER="${EVAL_PROVIDER:-openai}"
_INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_INFRA_SCRIPT="$_INFRA_DIR/_setup_infra-${EVAL_PROVIDER}.sh"

if [ ! -f "$_INFRA_SCRIPT" ]; then
  echo "ERROR: unknown EVAL_PROVIDER '$EVAL_PROVIDER' (no $_INFRA_SCRIPT)" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$_INFRA_SCRIPT"
