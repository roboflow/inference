# EditImageMetadata Workflow Block

A new Inference workflow sink block that writes metadata and tags back to existing Roboflow source images via the Image Metadata API. Designed to enable Asset Library batch processing: workflows can stamp model outputs (detected attributes, classifications, scores) onto images as filterable Firestore fields.

Paired with the platform-side REST endpoint design doc, `IMAGE_METADATA_API.md` (lives in the `roboflow` monorepo at `app/docs/IMAGE_METADATA_API.md`).

## Status

Design locked. Not yet implemented.

Target landing location: `inference/core/workflows/core_steps/sinks/roboflow/edit_image_metadata/v1.py`.

## Context

### Problem

Inference workflow blocks can read images and produce structured outputs — detection results, classifications, blur scores, dominant colors. Today there is no scalable way to write those outputs back to the source image as **filterable** attributes (Firestore `user_metadata`, `user_tags`).

The closest existing block, `RoboflowDatasetUploadBlockV1/V2`, **creates new images** in a dataset — it doesn't update existing ones. The Asset Library batch processing use case is the inverse: the images already exist, and workflows enrich them with metadata derived from CV models so users can filter (`color:red`, `blur_score>0.7`) in the Asset Library UI.

### Scope

- **In scope:** writing metadata key-value pairs and adding tags to **existing** source images.
- **Out of scope (v1):** removing metadata keys, removing tags, writing annotations, creating new images. The API supports `removeMetadata` / `removeTags` but the workflow editor doesn't expose them. They are reserved for direct API callers.

### Where it fits in the broader architecture

```
User selects images in Asset Library (or a RoboQL query)
    │
    ▼
User picks a workflow to run on those images
    │
    ▼
Backend creates a batchProcessing job (existing infrastructure)
    │
    ▼
Batch processing worker loads images + source IDs from JSONL
    │
    ▼
Worker invokes Inference workflow engine, batch-by-batch
    │
    ▼
Workflow runs: model blocks → ... → EditImageMetadata block
    │
    ▼
EditImageMetadata calls the platform's Image Metadata API
    │
    ▼
Platform writes to Firestore → ES re-indexes → image becomes filterable
```

## Block contract

### Manifest (user-configurable inputs)

| Field | Type | Required | Description |
|---|---|---|---|
| `source_id` | `Union[str, Selector(kind=[STRING_KIND])]` | yes | The Roboflow source ID (Firestore document ID) of the image to write to. Per-image batched input — also acts as the block's batch-dimensionality anchor. |
| `metadata` | `Optional[Union[Dict, Selector(kind=[DICTIONARY_KIND])]]` | no | Key-value metadata to set on the image. |
| `tags` | `Optional[Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]]` | no | Tags to add to the image. |
| `disable_sink` | `bool` | no | Kill switch. Defaults to `False`. |

Batched parameters (declared in `get_parameters_accepting_batches()`):
`["source_id", "metadata", "tags"]`. The block does not take an `images` input — image bytes are never sent to the API, and the workflow engine drives the batch dimension off `source_id`.

`metadata` and `tags` remain declared as batch-compatible inputs. Static values
or scalar selectors are expected to be auto-batched by the workflow engine when
the block is executed against a batch, so a single metadata dict or tag list can
be applied to every image in the batch.

Notes:
- **`removeMetadata` / `removeTags` are intentionally not exposed** in the manifest. The API supports them, but workflows are "stamp these values" use cases. Power users who need to remove fields can call the API directly.
- **No `fire_and_forget` flag.** Async behavior is determined automatically by batch size (see "Behavior" below), not by user toggle. The HTTP calls are fast enough that local fire-and-forget is unnecessary.
- **No quota gates** (unlike `DatasetUploadBlock`). Metadata writes are small Firestore updates with no per-customer cost model.

### Injected dependencies

```python
def __init__(self, api_key: Optional[str], cache: BaseCache):
    self._api_key = api_key
    self._cache = cache

@classmethod
def get_init_parameters(cls) -> List[str]:
    return ["api_key", "cache"]
```

- `api_key` — standard registered initializer. Comes from the engine context (env var, or explicit `init_parameters={"workflows_core.api_key": "..."}` on engine init).
- `cache` — used only for `get_workspace_name(api_key, cache)` lookup caching. Same pattern as `DatasetUploadBlock`.

The workspace is resolved at runtime from the API key via the existing `inference.core.roboflow_api.get_roboflow_workspace` helper — no manifest field needed. Roboflow API keys are workspace-scoped today, so the API key implies the target workspace.

### Outputs

Same shape as `RoboflowDatasetUploadBlockV1.run()`: a list of dicts, one per input image, in input order:

```python
[
    {"error_status": False, "message": "Metadata updated"},
    {"error_status": True,  "message": "Image not found in workspace"},
    ...
]
```

`describe_outputs()` returns:

```python
[
    OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
    OutputDefinition(name="message", kind=[STRING_KIND]),
]
```

## Behavior

### Endpoint selection: batch-size-driven

```
run(images: Batch[N], source_id, metadata, tags, ...)
    │
    ├── Build effective updates by skipping empty rows and merging duplicate source_id values
    │
    ├── M == 0  →  No HTTP request
    │              Return per-input skip results
    │
    ├── M == 1  →  POST /{workspace}/images/{source_id}/metadata   (sync)
    │              Real per-image result populated from response.
    │
    └── M > 1   →  POST /{workspace}/images/metadata               (async)
                   Body: { updates: [...] }
                   Server returns 202 + taskId immediately.
                   Block returns submission results for each original input
                   without waiting for completion.
```

`M` is the number of effective updates after empty rows are skipped and duplicate
`source_id` values are merged. This is the entirety of the "aggregation"
strategy. There is **no cross-call buffering** — each `run()` call is a
self-contained unit. If the workflow engine receives a batch of 100 images in
one `run_workflow()` call, the block fires at most one batch HTTP request. If the
caller streams images one at a time, the block fires N sync single-image
requests.

#### Why no buffering?

Cross-call buffering was considered (the `ModelMonitoringInferenceAggregatorBlockV1` pattern). Rejected because:

1. **Reliability:** flushing the final buffer in `__del__` is unreliable — Python's GC can run after the HTTP response is sent (no error surface), or never (if the process is killed mid-job). For a sink that writes user data to the platform, losing the final partial chunk is unacceptable.
2. **Error visibility:** errors thrown inside `__del__` have nowhere to surface. The workflow has already returned success.
3. **Locus of concern:** how many images to feed `run_workflow()` per invocation is the batch processing caller's decision (memory, GPU throughput, worker concurrency). The block shouldn't second-guess that decision.

If the batch processing infrastructure ends up passing single-image batches (`Batch[1]`) for a 10K-image job, the right fix is to batch images at the caller, not buffer at the block.

### Disabled sink

If `disable_sink=True`, returns immediately:

```python
[{"error_status": False, "message": "Sink was disabled by parameter `disable_sink`"} for _ in images]
```

Same pattern as `DatasetUploadBlock`.

### Missing API key

Raises `ValueError` at `run()` entry — same as `DatasetUploadBlock`:

```python
if self._api_key is None:
    raise ValueError("EditImageMetadata block requires Roboflow API key. ...")
```

### Empty updates

The block builds the `updates` list by zipping `source_id`, `metadata`, `tags`.
If a per-image field is `None` or missing, that bucket is omitted from the
update payload. If both buckets are empty for an image after normalization, the
block skips that image locally instead of sending an invalid empty update to the
API:

```python
{"error_status": False, "message": "Skipped because no metadata or tags were provided"}
```

Skipped images do not contribute to the single/batch endpoint decision. If every
input image is skipped, the block returns only skip results and performs no HTTP
request.

### Duplicate source IDs

Before calling the API, the block merges updates with duplicate
`source_id` values. The merge is defined to match sequential execution in input
order:

- Metadata key conflicts use **last write wins**.
- Repeated tags are de-duplicated; the resulting tag set is equivalent to
  sending the updates sequentially because the platform applies tags as an
  additive set.
- The returned block result still has one entry per original input image, in
  original order. All original entries that participated in a merged submitted
  update report the same submission result.

This avoids avoidable batch preflight failures such as duplicate `imageId` while
preserving the behavior users would get from one request per image.

### Batch size limit

The platform batch endpoint accepts at most 1000 updates. The block does **not**
chunk a single `run()` call into multiple requests. After skipping empty updates
and merging duplicate `source_id`s, if more than 1000 updates remain, the block
raises an error instructing the caller to reduce the workflow batch size.

### Per-image and API errors

The block does not validate metadata keys, value types, or tag formats — the API
endpoint owns that validation. Synchronous single-image API failures are returned
as the single image's `message`. Batch preflight/API failures are treated as
block errors rather than silently swallowed, because no async task is created and
there is no per-item result to poll. Per-item failures after async task creation
remain on the platform task's `failedItems` result.

## Comparison to `RoboflowDatasetUploadBlock`

| Aspect | `DatasetUploadBlock` | `EditImageMetadata` |
|---|---|---|
| Operation | Creates new image | Updates existing image |
| Endpoint | `POST /{workspace}/{dataset}/upload` (per image) | `POST /{workspace}/images/{id}/metadata` or `POST /{workspace}/images/metadata` |
| Payload | Multipart with image bytes | JSON, metadata/tags only |
| Per-image cost | Slow (compression, multipart, GCS write) | Fast (single Firestore update or PubSub publish) |
| Needs | API key, dataset name | API key, source ID |
| Quota gates | Yes — active-learning rate limits | No |
| `fire_and_forget` | Yes (manifest field, FastAPI BackgroundTasks / ThreadPoolExecutor) | No (server-side async handles it) |
| Output | `error_status`, `message` | `error_status`, `message` (same) |
| Iteration | Per-image HTTP in a loop | Per-image OR one batch HTTP |

## Implementation sketch

```python
from inference.core.roboflow_api import get_workspace_name

class EditImageMetadataBlockV1(WorkflowBlock):

    def __init__(self, api_key: Optional[str], cache: BaseCache):
        self._api_key = api_key
        self._cache = cache

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        source_id: Batch[str],
        metadata: Optional[Batch[Optional[Dict[str, Any]]]] = None,
        tags: Optional[Batch[Optional[List[str]]]] = None,
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return [
                {"error_status": False, "message": "Sink was disabled by parameter `disable_sink`"}
                for _ in images
            ]
        if self._api_key is None:
            raise ValueError("EditImageMetadata block requires Roboflow API key.")

        workspace = get_workspace_name(api_key=self._api_key, cache=self._cache)

        metadata = metadata or [None] * len(images)
        tags = tags or [None] * len(images)

        results = [None] * len(images)
        updates_by_source_id = {}
        source_id_to_result_indices = {}

        # Build per-image update payloads, skipping empty rows and merging
        # duplicate source IDs to match sequential last-write-wins behavior.
        for index, (sid, md, tg) in enumerate(zip(source_id, metadata, tags)):
            update = {"imageId": sid}
            if md:
                update["metadata"] = md
            if tg:
                update["addTags"] = tg
            if len(update) == 1:
                results[index] = {
                    "error_status": False,
                    "message": "Skipped because no metadata or tags were provided",
                }
                continue

            merged_update = updates_by_source_id.setdefault(sid, {"imageId": sid})
            if "metadata" in update:
                merged_update.setdefault("metadata", {}).update(update["metadata"])
            if "addTags" in update:
                merged_update.setdefault("addTags", [])
                for tag in update["addTags"]:
                    if tag in merged_update["addTags"]:
                        merged_update["addTags"].remove(tag)
                    merged_update["addTags"].append(tag)
            source_id_to_result_indices.setdefault(sid, []).append(index)

        updates = list(updates_by_source_id.values())
        if not updates:
            return results
        if len(updates) > 1000:
            raise ValueError("EditImageMetadata block supports at most 1000 updates per run().")

        if len(updates) == 1:
            submitted_result = _call_single_endpoint(
                workspace, updates[0], api_key=self._api_key
            )
        else:
            submitted_result = _call_batch_endpoint(
                workspace, updates, api_key=self._api_key
            )

        for update in updates:
            for index in source_id_to_result_indices[update["imageId"]]:
                results[index] = submitted_result
        return results
```

Helper functions (`_call_single_endpoint`, `_call_batch_endpoint`) live in `inference/core/roboflow_api.py` alongside the existing `register_image_at_roboflow`, `annotate_image_at_roboflow`, etc.

## Open questions (downstream of this doc)

These belong to the batch-processing-orchestration side, not the block itself:

- **How does `source_id` flow into the workflow?** The batch processing infrastructure stages JSONL files where each row references an image. Each row must include the Roboflow source ID alongside the image reference, and the workflow engine must route that ID to the block's `source_id` input. (Solving this is the "RoboQL → JSONL bridge" work in the broader project.)
- **Job lifecycle UX.** When a user kicks off a bulk metadata enrichment from the Asset Library, what do they see? Progress bar, cancellation, per-item error report? This is a separate UI / orchestration concern.
- **Workflow-author UX.** Should the workflow editor expose a higher-level "auto-tag" preset that wires up `EditImageMetadata` with sensible defaults, instead of asking the user to wire `source_id` from a workflow input every time?
