# Staging experiment ledger

This file is the reconciliation point between live staging experiments and
deployable Git state. Production changes are out of scope.

For every staging mutation, add a row before applying it. Update the same row
with measurements, disposition, and a Git/image reference before moving to the
next experiment. Do not leave useful live configuration represented only in a
`kubectl` command or shell history.

| ID | UTC window | Hypothesis | Live target and change | Git/image source | Revert | Result artifact | Disposition |
|---|---|---|---|---|---|---|---|
| _example_ | _2026-08-12T00:00Z_ | _Higher worker capacity exposes the GPU knee_ | _staging `video-proc` GPU pool: `MAX_CONCURRENT_JOBS=4` to `24`_ | _commit SHA; immutable image digest; rendered values SHA-256_ | _restore prior Deployment revision and verify ready replicas_ | _ignored `results/SUITE_ID/suite.json`; durable summary committed separately_ | _pending / keep / revert_ |

## Required evidence

- Record the cluster context, namespace, workload UID, prior revision, image
  digest, rendered configuration hash, and operator identity.
- Record the exact benchmark suite/run IDs and the Prometheus time window.
- Keep API keys, access tokens, source URLs containing credentials, and generated
  job files out of the ledger and Git.
- A successful experiment is not complete until its reusable code/config is
  committed and pushed. A rejected experiment is not complete until staging is
  reverted and the reason is recorded.
- Before a production proposal, verify the live staging workload can be recreated
  from the recorded commits and immutable images without manual patches.

