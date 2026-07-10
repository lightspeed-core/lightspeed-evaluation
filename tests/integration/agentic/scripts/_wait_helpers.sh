#!/usr/bin/env bash
# Poll helpers for scenario setup scripts — sourced, do NOT run directly.
#
# Each helper polls every 5s until the fixture's advertised fault state is
# actually observable (so the agent never evaluates a cluster that
# contradicts the proposal request), and fails loudly with a pod listing
# on timeout. Requires TEST_NS to be set by the sourcing script.

_poll() { # timeout_seconds description check_fn [args...]
  local timeout="$1" what="$2"
  shift 2
  local elapsed=0
  while [ "$elapsed" -lt "$timeout" ]; do
    if "$@"; then
      echo "Reached expected state after ${elapsed}s: $what"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "ERROR: timed out after ${timeout}s waiting for: $what" >&2
  oc get pods -n "$TEST_NS" -o wide >&2 || true
  return 1
}

_container_state_matches() { # app_label reason_regex
  oc get pods -l "app=$1" -n "$TEST_NS" -o jsonpath='{range .items[*].status.containerStatuses[*]}{.state.waiting.reason}{" "}{.state.terminated.reason}{" "}{.lastState.terminated.reason}{" "}{end}' 2>/dev/null \
    | grep -Eq "$2"
}

# Wait until a container of the labelled pods reports a waiting/terminated
# reason matching the regex (checks current and last state).
wait_for_container_state() { # app_label reason_regex [timeout=120]
  _poll "${3:-120}" "container of app=$1 in state matching '$2'" \
    _container_state_matches "$1" "$2"
}

_init_container_state_matches() { # app_label reason_regex
  oc get pods -l "app=$1" -n "$TEST_NS" -o jsonpath='{range .items[*].status.initContainerStatuses[*]}{.state.waiting.reason}{" "}{.state.terminated.reason}{" "}{.lastState.terminated.reason}{" "}{end}' 2>/dev/null \
    | grep -Eq "$2"
}

wait_for_init_container_state() { # app_label reason_regex [timeout=120]
  _poll "${3:-120}" "init container of app=$1 in state matching '$2'" \
    _init_container_state_matches "$1" "$2"
}

_pod_unschedulable() { # app_label
  oc get pods -l "app=$1" -n "$TEST_NS" -o jsonpath='{range .items[*].status.conditions[?(@.type=="PodScheduled")]}{.reason}{" "}{end}' 2>/dev/null \
    | grep -q "Unschedulable"
}

wait_for_unschedulable() { # app_label [timeout=120]
  _poll "${2:-120}" "a Pending pod of app=$1 marked Unschedulable" \
    _pod_unschedulable "$1"
}

_pod_evicted() { # app_label
  oc get pods -l "app=$1" -n "$TEST_NS" -o jsonpath='{range .items[*]}{.status.reason}{" "}{end}' 2>/dev/null \
    | grep -q "Evicted"
}

wait_for_evicted() { # app_label [timeout=210]
  _poll "${2:-210}" "an Evicted pod of app=$1" _pod_evicted "$1"
}

_pod_running_not_ready() { # app_label
  oc get pods -l "app=$1" -n "$TEST_NS" -o jsonpath='{range .items[*]}{.status.phase}{"/"}{.status.containerStatuses[0].ready}{" "}{end}' 2>/dev/null \
    | grep -q "Running/false"
}

wait_for_running_not_ready() { # app_label [timeout=120]
  _poll "${2:-120}" "a Running-but-not-Ready pod of app=$1" \
    _pod_running_not_ready "$1"
}

_no_pods_created() { # app_label
  # Sum desired replicas across ALL matching ReplicaSets so a stale
  # zero-scaled revision (e.g. from an interrupted earlier run) cannot
  # shadow the active one.
  local desired pods total=0 n
  desired=$(oc get rs -l "app=$1" -n "$TEST_NS" -o jsonpath='{.items[*].spec.replicas}' 2>/dev/null || true)
  for n in $desired; do total=$((total + n)); done
  pods=$(oc get pods -l "app=$1" -n "$TEST_NS" --no-headers 2>/dev/null | wc -l | tr -d ' ')
  [ "$total" -ge 1 ] && [ "$pods" -eq 0 ]
}

# For admission-rejected workloads (quota, LimitRange, PriorityClass): wait
# until the ReplicaSet exists but has created zero pods, then hold 10s to
# confirm the rejection persists rather than being creation lag.
wait_for_no_pods_created() { # app_label [timeout=60]
  _poll "${2:-60}" "ReplicaSet of app=$1 unable to create pods" \
    _no_pods_created "$1"
  sleep 10
  if ! _no_pods_created "$1"; then
    echo "ERROR: pods appeared for app=$1 after the initial rejection" >&2
    return 1
  fi
  echo "Confirmed: no pods created for app=$1."
}

# Server-side wait for a deployment to become Available, with the same
# on-timeout pod-listing diagnostics as the poll helpers.
wait_for_deployment_available() { # deployment [timeout=120]
  echo "Waiting for $1 deployment to become available..."
  if ! oc wait --for=condition=Available "deployment/$1" \
    -n "$TEST_NS" --timeout="${2:-120}s"; then
    echo "ERROR: deployment $1 did not become available within ${2:-120}s" >&2
    oc get pods -n "$TEST_NS" -o wide >&2 || true
    return 1
  fi
}

# Wait for MULTIPLE deployments under ONE shared time budget. A plain
# multi-resource `oc wait --timeout=Ns` applies the timeout PER RESOURCE
# sequentially (worst case N*timeout); this helper shares the deadline.
wait_for_deployments_available() { # timeout name...
  local timeout="$1" start elapsed remaining d
  shift
  start=$(date +%s)
  for d in "$@"; do
    elapsed=$(( $(date +%s) - start ))
    remaining=$(( timeout - elapsed ))
    if [ "$remaining" -lt 10 ]; then
      remaining=10
    fi
    wait_for_deployment_available "$d" "$remaining"
  done
}

_route_admitted() { # route
  oc get route "$1" -n "$TEST_NS" -o jsonpath='{.status.ingress[0].conditions[?(@.type=="Admitted")].status}' 2>/dev/null \
    | grep -q True
}

# kubectl/oc `wait --for=jsonpath` cannot express [?()] filters, so poll.
wait_for_route_admitted() { # route [timeout=60]
  _poll "${2:-60}" "route $1 admitted by the router" _route_admitted "$1"
}

_endpoints_populated() { # service
  oc get endpoints "$1" -n "$TEST_NS" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null \
    | grep -q .
}

wait_for_endpoints() { # service [timeout=120]
  _poll "${2:-120}" "service $1 to have endpoints" _endpoints_populated "$1"
}

# For selector-mismatch faults: hold briefly, then assert the service still
# has zero endpoints (its backing pods are Ready, so lag is not a factor).
confirm_no_endpoints() { # service
  sleep 5
  if _endpoints_populated "$1"; then
    echo "ERROR: service $1 unexpectedly has endpoints" >&2
    return 1
  fi
  echo "Confirmed: service $1 has no endpoints."
}

_pod_metrics_available() { # app_label
  oc adm top pods -l "app=$1" -n "$TEST_NS" --no-headers 2>/dev/null | grep -q .
}

# Wait until metrics-server reports usage for the labelled pods — for
# scenarios whose analysis depends on `oc adm top` data.
wait_for_pod_metrics() { # app_label [timeout=120]
  _poll "${2:-120}" "metrics for pods of app=$1" _pod_metrics_available "$1"
}

_logs_match() { # app_label regex
  oc logs -l "app=$1" -n "$TEST_NS" --tail=20 2>/dev/null | grep -Eq "$2"
}

# Wait until recent logs of the labelled pods match the regex — for faults
# whose only symptom is application-level (blocked traffic, DNS failures).
wait_for_log_match() { # app_label regex [timeout=120]
  _poll "${3:-120}" "logs of app=$1 matching '$2'" _logs_match "$1" "$2"
}

_logs_match_full() { # app_label regex
  oc logs -l "app=$1" -n "$TEST_NS" --tail=-1 2>/dev/null | grep -Eq "$2"
}

# Like wait_for_log_match but searches the ENTIRE log — for fixtures that
# replay history at startup, where the evidence scrolls past --tail=20.
wait_for_log_match_full() { # app_label regex [timeout=120]
  _poll "${3:-120}" "full logs of app=$1 matching '$2'" _logs_match_full "$1" "$2"
}

_job_failed() { # job_name
  oc get job "$1" -n "$TEST_NS" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null \
    | grep -q True
}

# Wait until a Job exhausts its backoffLimit and reports a Failed condition.
wait_for_job_failed() { # job_name [timeout=240]
  _poll "${2:-240}" "job $1 to report a Failed condition" _job_failed "$1"
}
