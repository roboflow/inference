# Repository instructions

Follow the shared repository guidance:

@AGENTS.md

## GitHub Actions

- Pin every action to a full commit SHA with an exact `# vX.Y.Z` comment;
  Dependabot keeps the pins current.
- Standard GitHub-hosted runners are free on this public repository; keep
  `ubuntu-latest`, macOS and Windows jobs there. Paid CPU jobs use Depot
  (`depot-ubuntu-22.04-*` or `depot-ubuntu-24.04-*` with `group: public-depot`),
  never GitHub larger runners.
- GPU jobs use `arc-gpu-runner-set`. Those runners are self-hosted: trigger GPU
  workflows only on `push`, `workflow_dispatch` or `workflow_call` from such a
  workflow, never on fork pull requests. Runner pods are ephemeral, so nothing
  persists on the runner between jobs.
- Every job sets `timeout-minutes`; a hung job otherwise bills six hours.
- Set `persist-credentials: false` on `actions/checkout` unless a later step
  pushes.
- Pass secrets and event data to `run:` through `env:`, never by interpolating
  `${{ }}` into the script.
- Grant `id-token: write` and other write permissions per job, only where used.
- Do not restore dependency caches in release or publish workflows.
- Validate workflow changes with `actionlint` (see `.github/actionlint.yaml`)
  and `uvx zizmor .github/workflows`.
