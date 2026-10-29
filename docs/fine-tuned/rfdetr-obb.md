# Native RF-DETR oriented object detection

This implementation adds the `oriented-object-detection` task for an exported
RF-DETR OBB model. It does not register the artifact as ordinary object detection.

## Artifact and geometry contract

The producer registers a standard RF-DETR ONNX package containing `weights.onnx`,
`class_names.txt`, and `inference_config.json`. The loader generates
`model_config.json` for its cache and offline loading. The registry identity is
architecture `rfdetr`, task `oriented-object-detection`, backend `onnx`, with model
variant `rfdetr-obb-preview-2xl`. The registry also accepts the producer's
`obb-detection` task alias; responses use `oriented-object-detection`.

The model metadata response supplies `orientedBoxMetadata` with a `classHeading`
map and nonnegative `classHeadingRevision`, taken from the model's frozen dataset
version. This field is optional for other tasks and required by the native OBB
loader. It follows the model through cached and offline loads. The runtime does
not infer class heading capability from the graph's heading head.

The graph emits raw `dets[B,Q,5]`, `labels[B,Q,C]`, and `heading[B,Q,4]` before
selection. `dets` contains normalized center and ordered side lengths, followed by
a clockwise-radian angle. `labels` contains class logits; output zero is background.
`class_names.txt` identifies output columns, and the package's standard
`class_names_operations` removes the background class. Response class IDs are
zero-based indices after that remapping, rather than COCO category IDs. Frozen
class heading policy is matched by class name. Zero-area padded query slots are
discarded before selection. Nonfinite graph values and missing heading outputs
are errors.

For a directed class, heading side `k` adds `k*pi/2` to the raw angle and swaps
width/height iff `k` is odd. This preserves the exact footprint and designates the
predicted front. Undirected classes retain the raw ordered rectangle axis.
No independent front vector is persisted.

The package declares letterbox resizing, RGB padding 114, division by 255 and
ImageNet normalization. For an image of width `w`, height `h` and square model
input size `S`, the runtime resizes to `round(w*S/max(w,h))` by
`round(h*S/max(w,h))`, with each dimension at least one pixel. Pillow bilinear
resizing and integer centered padding match the producer's training transform.

Rounding can make the realized width and height ratios differ. The inverse maps
centers using their respective ratios and ordered sides using the mean of those
two ratios, matching the training-target transform; angles remain unchanged.
The producer's current postprocessing uses a continuous max-side inverse.
Comparison with that postprocessing and validation against trained weights remain
pending. Roboflow static crop, grayscale and contrast preprocessing from
`inference_config.json` runs before model normalization. The inverse includes the
static-crop offset. Returned geometry remains floating point in original
input-image coordinates.

## Response and SDK

`POST /infer/oriented_object_detection` accepts the shared image/confidence request
fields. Each prediction contains `obb: [cx, cy, width, height, theta]`, `class_id`,
`class`, `confidence`, `heading`, and `detection_id`. The response includes
`task_type: "oriented-object-detection"` and frozen `class_heading` semantics.
Detection IDs use a fresh namespace for each input result, so repeated calls and
separate models produce distinct IDs. IDs remain stable through Workflow transport.
Bounding AABBs and corners are compatibility views derived from native geometry.

The current Inference SDK in API v1 mode routes the native task to this endpoint. It disables
client resizing for this task so a second rounded resize cannot alter the native
shape before the server's declared preprocessing. Select API v1 explicitly when connecting to a hosted URL that otherwise defaults to the legacy API v0 client; API v0 client downsizing is not part of the advertised native path.

## Workflows

`roboflow_core/roboflow_oriented_object_detection_model@v1` produces the separate
`oriented_object_detection_prediction` kind. Numeric confidence, class filtering,
row selection, row combination, dynamic crop offsets, original-coordinate output,
serialization, and `roboflow_core/oriented_box_visualization@v1` retain the ordered
footprint and class front. The visualization highlights the native front edge for
directed classes. NumPy and tensor carriers retain the same semantic fields.

Inside Workflows, ordered corners carry geometry/front through coordinate changes.
A general affine/perspective warp can produce a quadrilateral: serialization then
emits `geometry_type: "quadrilateral"` and ordered `corners`, without inventing an
OBB fit. Such results require an explicit rectangle-fit policy before use as native
rectangle annotations. Isotropic scaling and crop offsets preserve native tuples
without integer rounding.

The existing Detections Merge block fits a union AABB and is not compatible with
this native kind. Row combination concatenates detections without refitting them.
Query-language `DetectionsShift` and `DetectionsOffset` are also excluded and reject
native inputs at execution, because their geometry operations only update AABBs.
Opt-in `roboflow_core/detections_consensus@v2` compares native footprints using
rotated IoU and retains the highest-confidence contributor of the selected class,
including its ordered geometry, front and identity. It reuses v1 voting and class
selection, with configured confidence aggregation. Its geometry overlap code ports
Agis Kounelis's [PR #2420](https://github.com/roboflow/inference/pull/2420) at
`ee7fe5288255df1c8401877feda3ddaca0118eef`. Frozen class policies survive consensus
and combination, including empty outputs; incompatible policies or mixed tasks
raise errors. Ordinary tensor inputs remain unsupported by v2; v1 retains its
existing behavior.
Dataset writeback and annotation acceptance need the platform's native adapters and
workspace admission checks; ordinary AABB upload paths are not advertised for OBB.

## Validation and deployment matrix

| Target | Implementation and validation |
| --- | --- |
| Local CPU ONNX model package | Implemented; synthetic three-output ONNX fixtures use the standard producer package files and frozen class metadata; pinned producer letterbox pixels and target geometry tested across five image shapes |
| Server request consumer | Implemented; metadata retrieval, AutoModel loading, cache/offline reloads, native request lifecycle and original-coordinate response tested locally with synthetic fixtures |
| Current Inference SDK | Native route exercised through the request adapter and SDK with a synthetic ONNX fixture; its response passes native preview and evaluation parsers; no hosted request executed |
| Local Workflow model, transport and visualization | Implemented; NumPy/tensor semantic and coordinate tests |
| Query-language `DetectionsShift` / `DetectionsOffset` | Excluded; native NumPy/tensor inputs reject before geometry mutation |
| Hosted production serving | Requires compatible package releases, trained artifact registration and serving deployment; no hosted/cloud run claimed |
| CUDA, TensorRT, CoreML, browser inferencejs | Excluded from this initial backend registration |
| Legacy `roboflow-python` `version.model` | Excluded; current Inference SDK support does not add legacy task dispatch |

Synthetic fixtures establish package and transport compatibility; they do not
establish trained-model accuracy or an end-to-end platform training run. No model
package release, cloud/GPU run, deployment or feature activation is part of these
local validation results. Existing authorized model reads remain independent
of the platform's default-off gate for new OBB authoring/jobs.

## Package and worker release gate

This feature is pending release. The registry versions below are forward
requirements, not evidence of a published or deployed native provider:

| Distribution | Required version | Consumer |
| --- | --- | --- |
| `inference-sdk` | `1.8.4` | Remote native task dispatch in Workflows and model evaluation |
| `inference-models` | `0.39.7` | Native RF-DETR ONNX registry/runtime |
| `roboflow-workflows` | `0.2.15` | Canonical native response entities, blocks and consensus v2 |
| `inference`, `inference-cpu`, `inference-gpu`, `inference-core` | `1.8.4` | HTTP server and model adapters |

### Bootstrap before merging the consumer PR

The existing `publish.pypi.yml` and `publish.pypi.inference_exp.yml` workflows
already exist on the default branch, support `workflow_dispatch`, and were read
as active on 2026-10-27. GitHub requires the workflow to exist on the default
branch; `--ref` then selects its version on the reviewed feature branch. A new
workflow file existing only in a PR cannot be assumed dispatchable. See
[GitHub's manual workflow instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
and [`gh workflow run --ref`](https://cli.github.com/manual/gh_workflow_run).

The SDK stays owned by `publish.pypi.yml`, its existing publisher identity.
SDK-only mode uses `make create_sdk_wheel` and resolves no Workflows dependencies.
Normal release events still build and publish the SDK plus all root packages;
the SDK's separate upload skips an already-published bootstrap wheel while root
uploads retain duplicate-version checks. Manual runs default to build-only.

A maintainer must review and authorize branch publication separately. These
commands document that future operation; no dispatch or publication has been
performed by this change. Confirm the remote branch still points to the reviewed
commit before each run, and verify each run's `head_sha` and artifact metadata.
Build-only commands for the existing workflows are:

```bash
gh workflow run publish.pypi.yml --repo roboflow/inference \
  --ref caio/native-obb-consensus-current -f package=sdk -f publish=false
gh workflow run publish.pypi.inference_exp.yml --repo roboflow/inference \
  --ref caio/native-obb-consensus-current -f publish=false
```

After reviewing those build artifacts, a separately authorized maintainer can
repeat the same commands with `-f publish=true` to publish SDK `1.8.4` and models
`0.39.7`. The root workflow's SDK mode publishes only the isolated SDK wheel;
it cannot publish root distributions. The models workflow retains its independent
publisher. Both can run from this one reviewed branch before the consumer PR
merges, because their workflow files are already registered on the default
branch. No second active PR, new workflow identity or vendored package is needed.

For local build inspection from that reviewed checkout, the corresponding commands
are `python -m pip install setuptools wheel uv`, `make create_sdk_wheel`, and
`cd inference_models && uv build`. They produce distributions only; local builds
do not authenticate or upload to PyPI.

### Publisher configuration and release gates

Before an authorized publication, a PyPI project owner must verify each project's
**Manage → Publishing** Trusted Publisher settings. The expected GitHub owner is
`roboflow`, repository `inference`, and workflow filename `publish.pypi.yml` for
`inference-sdk` and the root distributions, `publish.pypi.inference_exp.yml` for
`inference-models`, and `publish.pypi.roboflow_workflows.yml` for
`roboflow-workflows`. The workflow filename is part of the trusted identity; a
renamed/new SDK workflow would require separate registration. These existing
jobs declare no GitHub environment, so any configured environment must match
before publishing; environment or branch protection may require additional
maintainer approval. Current private PyPI publisher settings were not inspected
or changed. See [PyPI's publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

1. Publish and verify SDK `1.8.4` and models `0.39.7` from the reviewed branch using
   the existing workflow identities above. Check registry wheel contents and
   metadata; publication is an irreversible maintainer operation, separate from
   approval to push or merge this PR.
2. After both wheels exist on the registry, run `cd workflows && uv lock`,
   commit the generated lock on this same PR, and verify a fresh constrained
   installation. Until then this consumer draft is blocked from merge/release:
   the retained lock records previous real releases. No fabricated registry
   hashes or vendored wheels substitute for this gate. The generated models lock
   must also pass `cd inference_models && uv lock --check`.
3. Merge the single consumer PR only after those checks, then publish Workflows
   `0.2.15` through its existing workflow. Verify the wheel contains
   `core_steps/common/oriented_entities.py`. `make create_workflows_wheel` is the
   local build; its existing publish workflow defaults to build-only on manual
   dispatch. Verify clean ordinary server imports with the released dependency set.
4. Publish the root `1.8.4` packages through the existing root workflow and build
   server images. `make create_wheels` builds the root release set, including the
   SDK; the workflow isolates its upload so the bootstrap version can be skipped.
   Full CPU/GPU builds explicitly install their checkout models wheel; slim builds
   resolve the new registry pin. Verify installed model registration and a CPU
   ONNX fixture in actual deployed images and US/EU targets.
5. Deploy compatible serving and queue workers before enabling platform native
   producers. Native custom Auto Label jobs use direct HTTP and require no queue
   SDK/Supervision upgrade. Unavailable endpoints or non-native responses fail the
   job without an ordinary detection fallback. Native Workflow Auto Label
   admission is a separate unsupported path.

Python Supervision's native parser/NMM feature is independent. Its development
version is outside the platform's existing `<0.31.0` cap; the server/Workflows
native carrier constructs compatible `Detections` directly and does not require
that upgrade. Existing ordinary/legacy task discriminators remain valid.
