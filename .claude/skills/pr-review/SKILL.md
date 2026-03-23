---
name: pr-review
description: Design review of GitHub pull requests. Evaluates requirements alignment, code quality, test coverage, and documentation. Use when asked to review a PR or /pr-review.
disable-model-invocation: true
---

# PR Review

Read-only design review. **CI owns linting, formatting, and test execution** — do not run quality checks unless explicitly requested.

**Repository:** `lightspeed-core/lightspeed-evaluation`

## Workflow

### 1. Fetch PR

```bash
git fetch upstream pull/<PR_NUMBER>/head:pr-review-<PR_NUMBER>
git checkout pr-review-<PR_NUMBER>
```

Get metadata via WebFetch:

- URL: `https://github.com/lightspeed-core/lightspeed-evaluation/pull/<PR_NUMBER>`
- Prompt: Extract PR title, description, author, labels, files changed

**Review entire PR (all commits):**

```bash
git log --oneline upstream/main..HEAD
git diff --stat upstream/main...HEAD
git diff upstream/main...HEAD
```

### 2. Requirements Alignment

Compare PR description claims vs actual diff changes. Flag gaps, scope creep, or missing pieces.

### 3. Code Review

Read changed files and evaluate:

| Dimension | Check For |
|-----------|-----------|
| **Logic** | Correctness, error handling, unnecessary complexity |
| **Modularity** | Clear boundaries, no duplication |
| **Minimalism** | Smallest change meeting requirements |
| **Layout** | New features → `src/lightspeed_evaluation/`; respect AGENTS.md constraints |

Only raise issues with concrete evidence.

### 4. Test Coverage

Read test files for new/changed behavior. Flag: missing cases, weak assertions, wrong test layer.

**Do not execute tests** unless requested.

### 5. Documentation

Check if changes need updates to `docs/`, `README.md`, or `AGENTS.md`.

### 6. Verify Issues

Read relevant files to confirm each issue. Cross-check automated review comments. Drop false positives.

### 7. Cleanup

Return to original branch:

```bash
git checkout -
```

## Output

### Chat Summary
Brief verdict: requirements fit, key findings, recommendation.

### Artifact: `wip/pr/review/<PR_NUMBER>/SUMMARY[N].md`

**Multiple reviews:** First → `SUMMARY.md`, subsequent → `SUMMARY1.md`, `SUMMARY2.md`, etc.

Check existing: `ls wip/pr/review/<PR_NUMBER>/SUMMARY*.md 2>/dev/null | wc -l`

**Content:**
- **Overview**: PR metadata (title, author, commits, files changed)
- **Requirements Alignment**: Stated goal vs delivered vs gaps
- **Issues by Severity**: Critical / Major / Minor with locations
- **Test Coverage**: Changes vs tests present vs gaps
- **Documentation**: Which areas updated/needed
- **Strengths**: What works well
- **Verdict**: Approve / Request Changes / Comment with rationale

## Severity

| Level | Examples |
|-------|----------|
| **Critical** | Security, data corruption, wrong requirements |
| **Major** | Fragile logic, poor modularity, missing critical tests/docs |
| **Minor** | Naming, small simplifications, optional docs |

## Notes

- If `upstream` missing: `git remote add upstream https://github.com/lightspeed-core/lightspeed-evaluation.git && git fetch upstream`
- Review complete PR changeset, not individual commits
