# Package selection and single-candidate loading

A process manager that makes memory admission decisions must know which package
it is admitting before the worker initializes the model. The loader now exposes
two building blocks for that integration:

- `negotiate_model_packages(...)` accepts authorized provider metadata and returns
  candidates in the existing preference order without initializing a model.
  `require_compatible_explicit_package=True` applies the same compatibility and
  request constraints when an exact package ID is supplied. An incompatible
  explicit candidate is rejected; another ID is never substituted.
- `load_model_package_candidate(...)` initializes exactly one candidate using the
  existing artifact download, integrity, materialization, and lifecycle handling.
  It does not catch initialization errors or attempt another package.

The ordinary `AutoModel.from_pretrained(...)` path continues to own its existing
fallback loop and explicit-ID behavior. The loop shares the single-candidate
operation, so the two paths do not maintain separate implementations of model
initialization.

## Intended manager integration

1. Obtain authorized metadata through the existing weights provider. Resolve the
   effective worker configuration and device before negotiation. Provider access,
   alias resolution, disabled backends, and existing cache-selection semantics
   remain responsibilities of that preparation path.
2. Negotiate candidates with the normal constraints. For an explicitly requested
   ID, use the strict compatibility option and retain only that candidate.
3. Resolve the selected candidate's memory template and ask the admission manager
   to admit a new dedicated worker. Missing coverage returns an unavailable result;
   it does not select another backend merely because that backend has a template.
4. Hand the selected metadata and effective settings to the worker. Revalidate
   compatibility against that worker's environment with the strict explicit-ID
   option; rejection must not substitute another candidate. Initialize through
   `load_model_package_candidate` only after admission.
5. On a failure that permits fallback, the manager confirms worker cleanup before
   estimating and admitting the next normally ranked candidate in a fresh worker.
   Preserve the request's attempt limit. Explicit-ID requests have no alternative
   package fallback. GPU reclamation and capacity reservations belong to the manager.

These primitives do not yet wire the manager/worker protocol, publish templates,
or implement an admission decision. The single-candidate function assumes its
caller has completed authorization, negotiation and admission; passing package
metadata alone is not proof of those checks. Do not expose it directly as a public
request route. Its dependency parameters preserve existing initialization behavior;
they do not expand M1's one-model-per-process scope to shared or composite models.

Disk-cached artifact bytes remain reusable through the existing loader. A cache
hit does not authorize a new process, and loading a model is not part of the
metadata-only negotiation operation. Offline admission is deferred separately.
