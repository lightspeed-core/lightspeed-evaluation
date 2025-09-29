#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Verify: Checking if namespace exists"

# Check if ols-test-ns namespace exists
if oc get ns ols-test-ns >/dev/null 2>&1; then
    echo "✅ Namespace ols-test-ns exists"
    exit 0
else
    echo "❌ Namespace ols-test-ns not found"
    exit 1
fi
