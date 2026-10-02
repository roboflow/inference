# Classifier image embeddings

The **Embedding Model** block returns a feature vector or raw logits from a
classifier. Select a workspace classification model version
or a pretrained alias: `resnet18`, `resnet34`, `resnet50`, or `resnet101`.
Pretrained aliases use the `inference-models` package registry; enable
`USE_INFERENCE_MODELS=True` on deployments using the legacy model loader.

Supported model families are Roboflow ResNet, ViT, and DINOv3 classifiers, including
single-label and multi-label models. The block preserves trained weights, pooling,
internal layer normalization, and the model's original image preprocessing. It
does not add L2 normalization. The `output_type` parameter selects:

- **Feature Vector** (`feature_vector`, the default): the input to the final linear
  classifier, with one value per feature.
- **Logits** (`logits`): the output of the final linear classifier, including its
  bias, before Softmax or Sigmoid, with one value per class in the model's class order.
  These values are not probabilities. Exports that already return logits retain
  that output; exports with a final activation have that activation removed.

Other classifier families, including ConvNeXt and YOLO classifiers, are not
supported by this initial release.

```json
{
  "version": "1.0",
  "inputs": [{"type": "InferenceImage", "name": "image"}],
  "steps": [{
    "type": "roboflow_core/embedding_model@v1",
    "name": "embedding",
    "data": "$inputs.image",
    "model_id": "resnet101",
    "output_type": "feature_vector"
  }],
  "outputs": [
    {"type": "JsonField", "name": "embedding", "selector": "$steps.embedding.embedding"},
    {"type": "JsonField", "name": "embedding_info", "selector": "$steps.embedding.embedding_info"}
  ]
}
```

`data` accepts images, batches, and crops such as `$steps.cropping.crops` or
`$steps.slicer.slices`. Its image input and flat `embedding` output match the
CLIP Embedding Model block. Text input is not supported. Existing Cosine Similarity
blocks can consume the output directly.

Both choices use the same `embedding` output port. `embedding_info` accompanies
each vector. It contains the resolved model version, output type,
feature definition, vector dimension, normalization, and embedding `space_id`.
The `space_id` incorporates preprocessing. The HTTP endpoint and SDK also return
effective preprocessing, backend implementation, and output precision. ONNX
metadata includes the recovered feature tensor, transformation version, and
source artifact SHA-256.
Build references and query vectors with the same
model, preprocessing, and feature definition. Equal dimensions do not establish
compatibility: ViT and DINOv3 can both return 768 dimensions in different spaces.
Feature vectors and logits have different `space_id` values even when their
dimensions happen to match; regenerate reference vectors when changing `output_type`.
Pin deployment precision when calibrating distance thresholds; `precision`
describes the returned tensor's dtype, not every backend's internal arithmetic.

Connect `$steps.embedding.embedding` directly to either input of the existing
Cosine Similarity block, for both feature vectors and logits. Compare vectors
from the same model and output mode; Cosine Similarity checks dimensions but does
not inspect `embedding_info` or enforce matching `space_id` values.

In tensor mode, local execution passes materialized image tensors and embedding
tensors directly through the `inference-models` adapter. The block moves or casts
vectors only when needed to match the Workflow tensor device and float32 dtype.
The legacy ONNX loader retains its CPU NumPy preprocessing and returns tensors
without converting vectors to Python lists. Remote execution and final JSON
outputs use the same serialized format as list mode.

The loader requests the `image_embeddings` capability and negotiates an ONNX,
PyTorch, or Hugging Face package for the **same model version**. Classification-only
serialized TensorRT engines are excluded. An explicitly pinned incompatible
package or local model fails rather than substituting different weights. ONNX
Runtime can accelerate the derived graph using TensorRT when that execution
provider is installed and enabled by deployment configuration; engine caches
are separate from the classifier's cache. GPU TensorRT execution requires
validation on the target deployment hardware.

Existing ONNX exports do not need retraining when their final linear head is
recoverable. Extraction recognizes supported `fc`, `classifier`, and
`linear_layer` weights and verifies the boundary by following the classification
output backward. It supports Gemm and MatMul/Add heads with optional score
activations. It preserves the complete backbone, including nested control-flow
captures. Unrecognized or shared heads fail explicitly. Derived ONNX artifacts
are cached separately by source contents, feature definition, and transform
version; the source classifier is preserved. The model cache must be writable.
For offline deployment, preload the model with `required_capabilities=["image_embeddings"]`
and the intended `output_type` before disconnecting. A cached classification-only
engine does not satisfy this requirement. Feature and logits graphs and their
TensorRT execution-provider caches are distinct.
Dynamic-batch legacy ONNX models obey `MAX_BATCH_SIZE` before preprocessing.
Fixed-batch models use their required batch size, padding the final batch and
discarding padded output rows.

Use the HTTP endpoint `POST /infer/embeddings` with `model_id`, `api_key`, and
`image` (one image or a list in the usual Inference image format). It returns
`embeddings`, `embedding_info`, `time`, and inference metadata. Standard image
preprocessing overrides are accepted and recorded in the metadata. Set
`output_type` to `feature_vector` or `logits`; omitting it selects feature vectors.

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient("http://localhost:9001", api_key="YOUR_API_KEY")
result = client.get_image_embeddings("product.jpg", model_id="my-project/3")
logits = client.get_image_embeddings("product.jpg", model_id="my-project/3", output_type="logits")
vector = result["embeddings"][0]
info = result["embedding_info"]

# The async API has the same arguments and result format.
# result = await client.get_image_embeddings_async("product.jpg", "my-project/3")
```

The SDK returns one result per image, unwrapping a single result, and retains
metadata across batches. Both SDK methods forward the four
`InferenceConfiguration.disable_preproc_*` overrides, including
`disable_preproc_auto_orientation`, which maps to the server's
`disable_preproc_auto_orient` field. These overrides affect the embedding space;
configure reference and query requests consistently.
Remote Workflow execution requires a server exposing
`/infer/embeddings`. Hosted routing and the workspace model-picker UI must ship
support for this block alongside the server; this repository exposes the block
schema and capability requirement but does not deploy those hosted services.
Release the updated `inference-models` and `roboflow-workflows` packages together
with the Inference server and SDK; older installed packages do not provide this
capability and its tensor execution port.

For direct model use:

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "my-project/3", api_key="YOUR_API_KEY",
    required_capabilities=["image_embeddings"],
    output_type="feature_vector",  # Or "logits"; selects which graph to preload.
)
features = model.embed_images(images, input_color_format="bgr")
# Direct calls can also override the output type, preparing that graph on first use.
logits = model.embed_images(images, output_type="logits", input_color_format="bgr")
```

Nearest-centroid scoring, versioned reference storage, tile aggregation, and
green/yellow/red thresholds remain downstream. The existing Identify Outliers
block uses a rolling baseline and does not reproduce fixed known-style centroids.
