# Changelog

## Unreleased

### Changed

- **Breaking (workload introspection wire format), schema version `1` -> `2`:** `Discovery.unknown_reasons`
  no longer holds `<code>:<context>` strings. Every entry is now a `DiscoveryProblem` object with a
  closed machine-readable `code` (`declaration_unavailable`, `declaration_failed`,
  `unresolved_selector`, `invalid_resource_identifier`, `opaque_remote_workflow`,
  `custom_python_internals_unknown`), a human-readable `description` and an open JSON `details` map
  documented per code (`node_id`, `declaration`, plus `block_type` / `field` / `selector` /
  `resource_type` where they apply). Consumers branch on `code` and read `details`; the description
  is display text and must not be parsed. `WorkflowIntrospection.schema_version` is therefore `"2"`;
  the workflow-definition version and the Execution Engine version are unchanged. Blocks returning a
  `Discovery` build reasons with the factory helpers in
  `roboflow_workflows.execution_engine.entities.workload`.

- `RuntimeRestriction` is now the single entity a block authors a restriction with, and it moved to
  `roboflow_workflows.execution_engine.entities.workload` (re-exported from
  `roboflow_workflows.prototypes.block`, so the import path and the class object are unchanged). It
  gained two DEFAULTED fields appended after the existing ones, so every positional and keyword
  constructor stays valid: `code` (a stable machine-readable identifier, default
  `generic_restriction` - two generic restrictions with different notes stay distinct) and
  `applies_to_configuration` (a `key == value` map describing the TARGET deployment). `to_dict()`,
  the editor payload, carries neither of them and is byte-identical to before, as is every
  `get_restrictions()` body and the catalog the editor receives.
- **Breaking (block authoring API):** `discover_portable_restrictions()` is removed. Blocks now
  declare restrictions by overriding the public `get_actual_restrictions()` directly - one hook,
  no separate authoring hook. The portable `RestrictionMetadata` a workload document carries is
  DERIVED from the authored `RuntimeRestriction` by `restriction_metadata_of()`; no block authors
  both. A block that declares nothing returns an empty complete `Discovery[RuntimeRestriction]`;
  a block that declares something passes the data through the
  `actual_restrictions_of(declared=..., node_id=..., ignore_environment_restrictions=...)` helper,
  which normalises it (`None` = unknown, a list = complete, a `Discovery` = explicit completeness)
  and applies the flag. A plugin subclassing a built-in extends what it inherits by calling
  `super().get_actual_restrictions(ignore_environment_restrictions=...)`. The shared
  `*_PORTABLE_RESTRICTION` presets are now derived from their legacy twin, and the preset constants
  lost their now-meaningless `_PORTABLE` infix.

### Added

- `WorkflowBlockManifest.get_actual_restrictions(*, ignore_environment_restrictions=False) ->
  Discovery[RuntimeRestriction]`: the public instance API for a step's restrictions, the counterpart
  of `get_actual_outputs()`. `True` is the PORTABLE view - no host evaluation at all, every
  conditional declaration returned with its condition intact (it does NOT drop
  environment-dependent restrictions). `False` is the HOST view - only the
  `applies_to_configuration` predicates are evaluated, against the parsed package configuration
  (never `os.environ`), and an entry that definitively does not apply here is removed. The runtime,
  input-mode and step-execution-mode axes are never evaluated or guessed, nothing is mutated and no
  condition is stripped. A configuration key this package cannot evaluate keeps its entry and makes
  the result incomplete, reporting the key NAMES only. A declaration the portable contract cannot
  express (blank/invalid `code`, an empty axis list, a blank configuration key) is reported as
  `declaration_failed` instead of being published as complete, and a hook that raises is sanitised
  the same way at the workload-builder boundary. The default body covers a block that never adopted
  the API: it calls `self.get_restrictions()` - the block's own legacy override through ordinary
  Python dispatch, or the inherited `[]` - and wraps whatever comes back as incomplete, with a
  `declaration_unavailable` reason carrying `details.source = "get_restrictions"`. A legacy getter
  may already have filtered its entries for the host that answered and the inherited default declares
  nothing, so neither a short list nor an empty one proves absence;
  `ignore_environment_restrictions=True` cannot undo filtering done inside the classmethod and does
  not make the fallback complete. With `False` the fallback is evaluated against this host like any
  other declaration. A block whose class-level list is complete and environment independent can wrap
  `self.get_restrictions()` explicitly through `actual_restrictions_of()` and state completeness
  itself.
- Workload introspection asks blocks for that portable view explicitly
  (`ignore_environment_restrictions=True`) and projects it onto the existing
  `Discovery[RestrictionMetadata]`. `WorkflowIntrospection.schema_version` stays `"2"` and the
  restriction payload on the wire is unchanged.
- Workload introspection: every `ModelSummary` now carries `steps_by_dimensionality`, the number
  of its `used_by_steps` compiled at each input depth (same reference-depth semantics and
  string-keyed wire encoding as the workflow-level histogram). Each referring step is counted
  once per model; the values always sum to the length of `used_by_steps`.

## `0.2.1`

### Fixed

- Expose native Qwen 3.8 VL 27B in Qwen VLM blocks v2–v4, including model discovery and thinking support.
- SpaceXAI object detection encodes PNG at compression level 9. OpenCV's default compression put large frames over xAI's 25MB upload limit.

### Added

- SpaceXAI block (`spacexai@v3`): `grok-4.7` model option with `low`/`medium`/`high`/`xhigh` reasoning effort. Detection reuses the Grok 4.5/4.6 prompt.

---

## `0.2.0`

### Added

- Python 3.13 support (`requires-python` is now `>=3.10,<3.14`).
- OpenAI v7 detection and instance-segmentation accept optional `output_classes`, keeping visual
  prompts in `classes` while constraining and decoding model output with stable labels.
- Compile-time workload introspection: `describe_workflow_workload()` reports graph connectivity,
  per-step work operations, portable restrictions, dependent resources and model inventory without
  initialising blocks, loading models or evaluating custom Python. Blocks declare these facts via
  `discover_work_operations()`, `discover_portable_restrictions()` and
  `discover_dependent_resources()` on `WorkflowBlockManifest`; all built-in blocks are annotated, including
  the Kafka Consumer and Kafka Producer enterprise sinks.

---

## `0.1.2`

### Fixed

- Webhook sink: SSRF hardening (destination validation, DNS pinning, redirect and proxy refusal) now applies only when `ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES=false`. The default restores the previous plain `requests` transport, so environment `HTTP_PROXY` / `HTTPS_PROXY` work again.

---

## `0.1.1`

### Fixed

- Accept Python string class names from NumPy object arrays in detection-property expressions, avoiding `.item()` errors after custom Python blocks.

### Added

- Kafka Consumer and Kafka Producer enterprise blocks; `confluent-kafka` and `aws-msk-iam-sasl-signer-python` join the `enterprise` extra.

### Changed

- Dropped unused dependencies: `opencv-contrib-python`, `requests-toolbelt`, and test extras `pytest-asyncio`, `pytest-timeout`, `aioresponses`.

---

## `0.1.0`

- Initial release: Workflows execution engine and block library extracted from `inference`.
