# Execution Engine changes

This file supplements the repository-root `AGENTS.md` for
`inference/core/workflows/execution_engine/`.

When a change affects workflow compilation or execution behavior—including parsing,
graph construction, inlining, dynamic-block assembly, scheduling, input validation,
runtime error handling, or Execution Engine v1 compile/run outcomes—add behavior tests
and a user-facing `What changed` entry under `## Unreleased` in the same PR.

The authoritative changelog is in the `roboflow/docs` repository at
`workflows/developer-guide/execution-engine-changelog.md`. Open a parallel docs PR.
Do not edit this repository's redirect stub and do not bump
`EXECUTION_ENGINE_V1_VERSION`; maintainers choose and apply release versions.

This requirement does not apply to comments, type-only refactors, formatting,
tests/docs that only mirror existing behavior, or ordinary `core_steps/` block changes
that require no Execution Engine modification.

An unreleased entry uses:

```markdown
## Unreleased

**What changed**

* **Short title** — One or two sentences describing compile/run impact for workflow authors.
```
