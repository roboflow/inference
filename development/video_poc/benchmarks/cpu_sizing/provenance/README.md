# CPU process-image provenance

Before rendering any process-topology CPU patch, add and commit one JSON record
in this directory with:

- `schemaVersion: 1` and `environment: staging`;
- the resolved immutable `image` digest;
- the exact observed immutable CPU `baseImage`;
- the exact 40-character `sourceRevision`, which the renderer verifies is a
  descendant of the bounded-cleanup gate;
- the Cloud Build and credential-free smoke build IDs; and
- `smokePassed: true` only after the smoke completes successfully.

The renderer rejects untracked records and the thread-only baseline digest.
After rollout, the separate CPU process-containment gate must still pass before
capacity testing; build provenance is not runtime proof.
