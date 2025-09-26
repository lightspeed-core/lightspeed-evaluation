#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setup: Preparing test environment"

# Delete test namespace if it exists (clean start)
echo "Ensuring clean state: deleting ols-test-ns if it exists"
oc delete ns ols-test-ns --ignore-not-found

echo "✅ Setup completed successfully"
