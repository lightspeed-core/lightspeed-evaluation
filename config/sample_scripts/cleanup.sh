#!/usr/bin/env bash
set -euo pipefail

echo "🧹 Cleanup: Removing test environment"

# Delete test namespace
echo "Deleting test namespace: ols-test-ns"
oc delete ns ols-test-ns --ignore-not-found

echo "✅ Cleanup completed successfully"
