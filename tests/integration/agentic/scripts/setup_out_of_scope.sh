#!/usr/bin/env bash
set -euo pipefail

# Deploy provider infrastructure only — this scenario tests refusal of an out-of-scope namespace, so no fixture exists.
# Provider is selected via EVAL_PROVIDER — see _setup_infra.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_infra.sh
source "$SCRIPT_DIR/_setup_infra.sh"
# shellcheck source=_wait_helpers.sh
source "$SCRIPT_DIR/_wait_helpers.sh"

# No fixture: the request targets a namespace outside targetNamespaces.
echo "Setup complete."
