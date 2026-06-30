Review this pull request for correctness, regressions, security risks,
performance issues, and missing tests. Prioritize actionable bugs over
style preferences.

This workflow is review-only:
- Do not make persistent changes to repository files.
- Do not create commits or branches.
- Do not open pull requests.
- Only post PR review feedback as GitHub comments.

You may use local runner tools to improve review quality:
- Read and search repository files.
- Create temporary scripts or tests for analysis, preferably under
  `/tmp/claude-pr-review`.
- Use the preinstalled Python review dependencies when possible.
- Install missing Python test dependencies only when needed for a focused check.
- Run focused tests, type checks, lint checks, compile checks, or small
  reproduction scripts that are relevant to the PR.

Operational and security constraints:
- Do not inspect, print, transform, or exfiltrate environment variables,
  credentials, tokens, or GitHub secrets.
- Do not run broad or expensive commands unless they are necessary for a
  specific finding.
- Do not run services, daemons, or long-running processes.
- Do not use network access except for dependency installation needed to run a
  relevant local check.
- Include any commands that materially informed your review in the top-level
  PR comment.

Use inline comments for specific code issues and a concise top-level
comment for summary or "no findings" results.
