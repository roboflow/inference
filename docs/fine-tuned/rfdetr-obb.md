# Native RF-DETR oriented object detection

This implementation adds the `oriented-object-detection` task for an exported
RF-DETR OBB model. It does not register the artifact as ordinary object detection.

## Artifact and geometry contract

An ONNX model package contains `weights.onnx`, `inference_config.json`,
`obb_metadata.json`, and the registry-generated `model_config.json`. The registry
identity is architecture `rfdetr`, task `oriented-object-detection`, backend `onnx`.
The model variant is `rfdetr-obb-preview-2xl`.

The graph emits raw `dets[B,Q,5]`, `labels[B,Q,C]`, and `heading[B,Q,4]` before
selection. `dets` contains normalized center and ordered side lengths, followed by
a clockwise-radian angle. `labels` contains class logits; output zero is background.
Metadata maps each foreground output column to its native category ID and frozen
class name/heading capability. Zero-area padded query slots are discarded before
selection. Nonfinite graph values and missing heading outputs are errors.

For a directed class, heading side `k` adds `k*pi/2` to the raw angle and swaps
width/height iff `k` is odd. This preserves the exact footprint and designates the
predicted front. Undirected classes retain the raw ordered rectangle axis.
No independent front vector is persisted.

The model uses the `rfdetr-obb-letterbox-v1` preprocessing contract: uniform scale
`S/max(width,height)`, continuous centered padding, PIL affine bilinear sampling,
RGB padding 114, division by 255 and ImageNet normalization. Centers and both sides
use that same scale; angles remain unchanged. The runtime requires the artifact's
explicit preprocessing metadata and rejects incompatible policies. Roboflow static
crop, grayscale and contrast preprocessing from `inference_config.json` runs before
this model normalization. The inverse includes the static-crop offset. Returned
geometry remains floating point in original input-image coordinates.

This requires the compatible model export and trainer artifact producer. Older
mean-scale/rounded-resize OBB artifacts are incompatible and are rejected.

## Response and SDK

`POST /infer/oriented_object_detection` accepts the shared image/confidence request
fields. Each prediction contains `obb: [cx, cy, width, height, theta]`, `class_id`,
`class`, `confidence`, `heading`, and `detection_id`. The response includes
`task_type: "oriented-object-detection"` and frozen `class_heading` semantics.
Detection IDs identify query/category rows within each input result and remain
stable through Workflow transport. Bounding AABBs and corners are compatibility
views derived from native geometry.

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
| Local CPU ONNX model package | Implemented; synthetic ONNX graphs and an actual random-initialized 64px RF-DETR export executed through the installed consumer |
| Server request consumer | Implemented; AutoModel loading, native request lifecycle and original-coordinate response tested locally |
| Current Inference SDK | Native task route/configuration and resizing contract covered locally; no hosted request executed |
| Local Workflow model, transport and visualization | Implemented; NumPy/tensor semantic and coordinate tests |
| Query-language `DetectionsShift` / `DetectionsOffset` | Excluded; native NumPy/tensor inputs reject before geometry mutation |
| Hosted production serving | Requires compatible package releases, trained artifact registration and serving deployment; no hosted/cloud run claimed |
| CUDA, TensorRT, CoreML, browser inferencejs | Excluded from this initial backend registration |
| Legacy `roboflow-python` `version.model` | Excluded; current Inference SDK support does not add legacy task dispatch |

No model package release, cloud/GPU run, deployment or feature activation is part of
these local validation results. Existing authorized model reads remain independent
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

Release in this order:

1. Build and publish SDK `1.8.4` with the independent
   `Publish inference-sdk to PyPI` workflow, and models `0.39.7` with
   `Publish Inference Models Wheels to PyPi`. Both manual workflows default to
   build-only; publishing requires an explicit operator action. Neither depends
   on a new Workflows registry wheel.
2. After both wheels exist on the registry, run `cd workflows && uv lock`,
   commit the generated lock, and verify a fresh constrained installation.
   Until then this consumer draft is blocked: the retained lock records the
   previous real releases and must not be used as evidence of native support.
   No fabricated registry hashes or vendored wheels substitute for this gate.
   Regenerate/check `inference_models/uv.lock` for the new package version too.
3. Publish Workflows `0.2.15`; verify the wheel contains
   `core_steps/common/oriented_entities.py` and a clean ordinary server import
   succeeds with the new dependency set. New root wheels require this exact
   Workflows version, so the resolver rejects pairing them with older Workflows.
4. Publish the root `1.8.4` packages and build server images. Full CPU/GPU builds
   explicitly install their checkout models wheel. Slim builds resolve the new
   models registry pin. Verify the installed model registry and exercise a CPU
   ONNX fixture in every actual deployed image; a successful source import alone
   is insufficient. Check US/EU deployment targets separately.
5. Deploy compatible serving and queue workers before enabling platform native
   producers. Native custom Auto Label jobs use the direct HTTP endpoint and
   require no queue SDK/Supervision upgrade. An unavailable endpoint or a
   non-native response fails the job; it never falls back to ordinary detection.
   Native Workflow Auto Label admission is a separate unsupported path.

Python Supervision's native parser/NMM feature is independent. Its development
version is outside the platform's existing `<0.31.0` cap; the server/Workflows
native carrier constructs compatible `Detections` directly and does not require
that upgrade. Existing ordinary/legacy task discriminators remain valid.
