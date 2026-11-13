# Memory template admission (M1 draft)

This package implements the local C06–C16 matching and publication boundary.
It is not enabled automatically and does not download weights or start profiling.
Use an authorized server-side family adapter to construct `Request`, then call
`resolve(store.read_active(deployment_group), request=request)`.

## Supported first slice

- Separate one-model process, closed structural fields and known behavior keys.
- Exact GPU-level environment matching; broader cross-GPU claims are deferred.
- Finite explicitly validated **joint** workload points. No interpolation,
  monotonicity assumption or Cartesian product of independent maxima.
- Published priority order chooses designated template versions. The default band
  is resolved independently; hints only add required points. The result is the
  larger of the default and covering estimates and retains both evidence sets.
- Used-library versions match exactly unless an explicit reviewed directional
  range/operator permits transfer. Lists support exceptions. Denies win.
  Prereleases are included in explicit ranges; same-minor patch operators exclude
  prereleases. No PyTorch compatibility rule is enabled by default.
- inference-models version/build coverage is observe-only by default. Unknown
  coverage adds a limitation. Unknown behavior and known incompatibility reject.

## Firestore

Install the `memory-profiles` extra. Pass an authenticated client and its explicit
project to `FirestoreCatalog`; the reader never chooses a project from a request.
Templates use `memory_profile_templates/{id}/versions/{version}`. Catalogs embed
bounded envelope/family definitions, and pin templates by content hash. Policies
are immutable `memory_profile_policies/{id}` documents. The embedded definitions
are the first physical-layout implementation of the provisional catalog-owned
layout; they can move to subcollections if document limits require it.

`stage(snapshot)` creates immutable objects and a draft in one transaction.
Only the evidence-validated publisher should call it. `mark_ready(id, approval=...)`
requires the operator's review reference; credentials/Firestore IAM must restrict
all writer operations to authorized publication tooling. It does not grant access
or infer approval. `activate(..., expected_revision=...)` switches the catalog and
policy together with compare-and-swap. `withdraw(id)` prevents future reads and
cannot be undone by restaging/marking ready. Each lookup uses a Firestore read-only
transaction, including readiness and the active control revision. No document or
estimate cache is introduced. Original GCS evidence remains immutable.

Storage hashes use explicitly named `canonical-json/v1` (sorted compact UTF-8
JSON), not the producer measurement identity algorithm. Evidence `record_sha256`
uses RFC8785 in the publisher. Catalog definitions and selected template contents
are revalidated on read; GCS files are validated at publication, not every lookup.

## Rollout boundary

The actual RF-DETR descriptor and effective-setting capture mappings must be
reviewed against the pinned worker/runtime. Unknown facts must fail normalization.
This draft provides no inferred descriptor from an alias and no measured catalog.
The full campaign and its matrix remain pending user review. Test fixtures are
synthetic and must never be published as measurements. Physical JSON uses Python
snake_case storage fields; success responses retain the documented C15 names.
Broader environment claims, interval containment, composites and offline snapshots
require separate implementations and are rejected by this M1 subset.
