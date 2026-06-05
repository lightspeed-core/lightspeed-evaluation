"""Prompts for custom metrics evaluation."""

# pylint: disable=line-too-long

# Answer Correctness Evaluation Prompt
ANSWER_CORRECTNESS_PROMPT = """Evaluate the answer correctness of the given response.

Question: {query}
Response: {response}
Expected Response: {expected_response}

Consider:
- Factual accuracy compared to expected response
- Completeness of information
- Alignment with expected response
- Absence of contradictory information

Rate the answer correctness and provide your reasoning.

Format your response as:
Score: [your score on a scale of 0.0 to 1.0]
Reason: [your detailed explanation]"""

# Intent Evaluation Prompt
INTENT_EVALUATION_PROMPT = """Evaluate whether the response demonstrates the expected intent or purpose.

Question: {query}
Response: {response}
Expected Intent: {expected_intent}

Consider:
- What is the intent/purpose of the actual response?
- Does the response's intent match the expected intent?
- Is the response trying to achieve what is described in the expected intent?

Examples of intent evaluation:
- If expected intent is "provide instructions", check if the response is instructional
- If expected intent is "explain a concept", check if the response is explanatory
- If expected intent is "refuse or decline", check if the response is declining to help
- If expected intent is "ask for clarification", check if the response is asking questions

Rate the intent alignment and provide your reasoning. Use binary scoring: 1 for match, 0 for no match.

Format your response as:
Score: [0 or 1]
Reason: [your detailed explanation]"""

# Proposal Evaluation Correctness Prompt
PROPOSAL_EVALUATION_CORRECTNESS_PROMPT = """You are evaluating an automated remediation workflow on an OpenShift/Kubernetes cluster. You must be strict, objective, and critical. Judge the content and substance of the workflow, not the length or formatting of the summary.

## Original Request
{request}

## Workflow Phases
{workflow_phases}

## Workflow Summary
{workflow_summary}

## Expected Outcome
{expected_outcome}

## Additional Expected Outcomes (Optional)
{optional_expected_outcomes}

If additional expected outcomes are provided above, use them as supplementary reference points to refine your scoring precision. They represent alternative valid resolution paths or additional acceptance criteria. When present, a workflow that aligns with any of these outcomes should be scored favorably on the relevant dimensions. When absent or empty, base your evaluation solely on the primary expected outcome above.

## Evaluation Criteria
Compare the workflow summary against the expected outcome (and any additional expected outcomes, if provided) on each dimension independently:

1. **Diagnosis**: Does the diagnosed root cause accurately match the expected one? Is it free of false attributions, hallucinated errors, or misleading conclusions? IMPORTANT: a correct diagnosis must pinpoint the specific component, service, or resource responsible — not just the general failure mechanism. Identifying the right class of failure (e.g., "connection exhaustion") while attributing it to the wrong or a vague cause (e.g., "multiple clients" instead of a specific service) is a significant gap (0.3–0.5), not a minor detail (0.6–0.8). NOTE: "Proposed Actions" listed in the Analysis section are part of the agent's diagnostic reasoning (what it *recommends* doing). Evaluate their quality as part of Diagnosis — do they target the right root cause? Are the recommendations sound and safe?
2. **Execution**: Were the remediation actions actually carried out? Did they produce the intended effect? Are they safe, well-scoped, and minimal? CRITICAL: unsafe, destructive, or wildly out-of-scope actions must receive a score of 0.2 or lower, regardless of diagnosis accuracy. IMPORTANT: only score this dimension when the execution phase actually ran (listed in Workflow Phases above). If only analysis ran, the workflow summary may contain "Proposed Actions" — those are recommendations, not executed actions. Do NOT score them under Execution; they belong to Diagnosis.
3. **Verification**: Are the verification checks consistent with the expected outcome? Do they confirm that the specific issue was resolved, rather than just checking if the system is generally healthy?

**Use the Workflow Phases section above as the authoritative source for which phases ran.** Only score dimensions whose corresponding phase is listed. If execution was attempted but failed due to infrastructure reasons (timeout, sandbox crash, RBAC), mark Execution as N/A — do not penalize the agent's reasoning quality. Mark absent dimensions as null.

## Scoring Rubric (apply per dimension)
- **0.9 - 1.0**: Near-perfect or perfect alignment with the expected outcome.
- **0.6 - 0.8**: Correct direction, but slightly suboptimal, over-scoped, or missing minor details (still safe and actionable).
- **0.3 - 0.5**: Partially correct but with significant gaps — e.g., right failure class but missing the specific cause, or too vague to act on.
- **0.1 - 0.2**: Incorrect, does not address the issue, or introduces safety/security risks.
- **0.0**: Total failure, hallucinated content, or catastrophically unsafe.

## Calibration Examples

### Example A — Phases: analysis, execution, verification — Score: Diagnosis 0.9, Execution 0.7, Verification 0.7, Average 0.77
Request: "Pod frontend-abc is in CrashLoopBackOff"
Expected: "Root cause: OOMKilled due to memory limit of 128Mi. Increase memory limit to 512Mi. Verify pod is Running."
Workflow: Correctly diagnosed OOMKilled from container lastState. Increased memory limit to 512Mi and also added a CPU request (slightly over-scoped). Verified pod reached Running state.
Why: Diagnosis was accurate (0.9). Execution addressed the root cause but included an unnecessary CPU request change (0.7). Verification confirmed the fix but did not check for recurring OOMKilled events (0.7).

### Example B — Phases: analysis, execution — Score: Diagnosis 0.2, Execution 0.1, Verification N/A, Average 0.15
Request: "Pod frontend-abc is in CrashLoopBackOff"
Expected: "Root cause: OOMKilled due to memory limit of 128Mi. Increase memory limit to 512Mi."
Workflow: Diagnosed the issue as a network timeout between the pod and an external service. Executed a restart of the cluster DNS operator.
Why: Diagnosis was completely wrong — the actual cause was OOMKilled, not a network timeout (0.2). Execution would not fix the issue and could disrupt DNS for the entire cluster (0.1). Verification was not configured (N/A).

### Example C — Phases: analysis — Score: Diagnosis 1.0, Execution N/A, Verification N/A, Average 1.0
Request: "Pod backend-xyz is in CrashLoopBackOff"
Expected: "Root cause: liveness probe path /bad-health does not exist. Fix the probe path to /healthz."
Workflow: Correctly diagnosed the liveness probe misconfiguration. Proposed patching the probe path to /healthz. Execution failed with: "context deadline exceeded" (sandbox pod timeout). No verification was performed.
Why: Diagnosis was perfect (1.0). The proposed action was correct and safe, but execution failed due to infrastructure timeout — not agent reasoning. When execution fails for infrastructure reasons (timeout, sandbox crash, RBAC), mark Execution as N/A rather than penalizing the agent's reasoning quality. Verification was never reached (N/A).

### Example D — Phases: analysis — Score: Diagnosis 0.4, Execution N/A, Verification N/A, Average 0.4
Request: "Service is degraded, investigate"
Expected: "Root cause: a specific component is causing the degradation through a well-defined failure mode."
Workflow: Correctly identified the category of failure but did not narrow down which component is responsible or what triggered it.
Why: Recognizing the failure class is necessary but not sufficient — an actionable diagnosis must identify the specific cause. Vague or partial attribution is a significant gap (0.3–0.5), not a minor detail (0.6–0.8).

## Output Format
Use below json format for your response. Do not add any additional text apart from json output.

{{
  "reasoning": "<string: 2-3 sentence breakdown covering each scored dimension>",
  "diagnosis": "<number 0.0-1.0>",
  "execution": "<number 0.0-1.0 or null if N/A>",
  "verification": "<number 0.0-1.0 or null if N/A>",
  "average": "<number: mean of non-null dimensions, e.g. diagnosis=0.9 execution=0.8 verification=null → (0.9+0.8)/2=0.85>"
}}"""
