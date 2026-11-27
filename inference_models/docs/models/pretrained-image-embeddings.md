# DINOv2, DINOv3 and SigLIP2 - Image Embeddings

Pretrained vision backbones that turn an image into a single feature vector, for similarity search, clustering, deduplication and other work on visual similarity. One implementation, `PretrainedImageEmbeddingOnnx`, serves all three model families from ONNX packages.

## Overview

- **Image Embeddings** - One vector per image, from a pretrained backbone with no task head
- **Image-only** - These packages hold no text encoder, so they do not compare images with text
- **Raw vectors** - Embeddings are returned as the model produces them, **not normalised**. L2-normalise them before cosine similarity

!!! info "License & Attribution"
    **DINOv2**: Apache 2.0, [Meta DINOv2](https://github.com/facebookresearch/dinov2)<br>**SigLIP2**: Apache 2.0, [Google SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224)<br>**DINOv3**: [DINOv3 License](https://github.com/roboflow/inference/blob/main/inference_models/inference_models/models/dinov3/LICENSE.txt), [Meta DINOv3](https://github.com/facebookresearch/dinov3)

!!! warning "DINOv3 License"
    DINOv3 is released under Meta's DINOv3 License, the same as the [DINOv3 classifier](dinov3-classification.md). It allows research and commercial use but restricts some uses (trade controls, military and weapons applications, among others), requires acknowledging DINO Materials in publications and requires redistributing under the same terms. Meta's `facebook/dinov3-*` repositories are gated on Hugging Face; the onnx-community export this package uses is not, and its weights remain under the DINOv3 License.

## Model IDs

| Model ID | Vector | Dimension | Input | Source |
|----------|--------|-----------|-------|--------|
| `dinov2/small` | CLS token of `last_hidden_state` | 384 | 224×224 | [`onnx-community/dinov2-small`](https://huggingface.co/onnx-community/dinov2-small), fp32 graph |
| `dinov3/vits16` | CLS token of `last_hidden_state` | 384 | 224×224 | [`onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX), fp32 graph with external weights |
| `siglip2/base-patch16-224` | `pooler_output` of the vision model | 768 | 224×224 | [`onnx-community/siglip2-base-patch16-224-ONNX`](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX), fp32 graph |

Each model's `embedding_info` names its vector with a versioned feature definition: `dinov2-cls-token@v1`, `dinov3-cls-token@v1` and `siglip2-pooler-output@v1`. Vectors with different feature definitions live in different spaces and must not be compared.

Packages are built from pinned Hugging Face revisions with the developer script described in [Building Packages](#building-packages). Until they are published by a weights provider, load them from a local directory.

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu118`, `onnx-cu12`, `onnx-jp6-cu126` |

## Package Layout

| File | Content |
|------|---------|
| `model.onnx` | The graph, taking `pixel_values` as a float32 NCHW batch with a dynamic batch size |
| `model.onnx_data` | External weights, only for graphs that have them (DINOv3). Loaded from next to `model.onnx` |
| `preprocessor_config.json` | Verbatim Hugging Face image processor config |
| `embedding_config.json` | Which output holds the vector, and how to read it |
| `model_config.json` | `model_architecture` (`dinov2`, `dinov3` or `siglip2`), `task_type` `embedding` and `backend_type` `onnx`, for local loading |

`embedding_config.json`:

```json
{
  "output_name": "last_hidden_state",
  "token_index": 0,
  "dimension": 384,
  "feature_definition": "dinov2-cls-token@v1",
  "source": {
    "repository": "onnx-community/dinov2-small",
    "revision": "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d",
    "file": "onnx/model.onnx"
  }
}
```

With `token_index` set, the output is read as `[batch, tokens, dimension]` and that token is taken. With `token_index` `null`, the output must already be `[batch, dimension]`. Loading fails when the graph lacks the output, its rank does not fit `token_index` or its dimension differs from `dimension` (for a graph that leaves the dimension dynamic, the first inference fails instead).

## Preprocessing

Preprocessing follows the package's `preprocessor_config.json`, implemented with torch on the model's device. The supported subset of the Hugging Face image processors (`BitImageProcessor`, `SiglipImageProcessor`, `DINOv3ViTImageProcessor`):

- `do_resize` with `size` `{"shortest_edge": N}` (shorter side to N, aspect ratio kept) or `{"height": H, "width": W}` (stretched)
- `resample` `2` (bilinear) or `3` (bicubic), antialiased
- `do_center_crop` with `crop_size`
- `do_rescale` with `rescale_factor`, and `do_normalize` with `image_mean` and `image_std`

Steps run in the order of the declared processor: DINOv3's processor rescales before resizing, the others resize first. Any other key or value is rejected when the package loads, and preprocessing must produce a fixed image size.

| Model ID | Resize | Crop | Mean / Std |
|----------|--------|------|------------|
| `dinov2/small` | Shorter side to 256, bicubic | Center 224×224 | ImageNet |
| `dinov3/vits16` | Stretch to 224×224, bilinear | None | ImageNet |
| `siglip2/base-patch16-224` | Stretch to 224×224, bilinear | None | 0.5 / 0.5 |

Output matches the torchvision image processors of `transformers` up to float rounding. Their PIL processors differ by at most two 8-bit levels on rare pixels.

Numpy images are read as BGR and tensors as RGB unless `input_color_format` says otherwise. The effective preprocessing is available as `model.embedding_preprocessing`.

## Usage Examples

### Image Embeddings

```python
import cv2
import torch
from inference_models import AutoModel

model = AutoModel.from_pretrained("/path/to/packages/dinov2/small")
images = [cv2.imread("image1.jpg"), cv2.imread("image2.jpg")]

embeddings = model.embed_images(images)  # float32, shape (2, 384)
print(model.embedding_info)
# {'feature_definition': 'dinov2-cls-token@v1', 'output_type': 'feature_vector', 'normalization': 'none'}
```

### Similarity

```python
import torch.nn.functional as F

normalized = F.normalize(embeddings, dim=1)
similarity = normalized[0] @ normalized[1]
```

Only `feature_vector` embeddings are available. Requesting another `output_type`, such as `logits`, raises `ModelInputError`.

## Building Packages

`inference_models/development/packaging/build_pretrained_image_embedding_packages.py` downloads the pinned files, writes the package layout above, checks the graph outputs, external data and preprocessing, loads the package to embed one image, and prints each file's size and sha256:

```bash
python inference_models/development/packaging/build_pretrained_image_embedding_packages.py \
    --output-dir ./packages
```

Pass `--models` to build only some of them, and `--onnx-execution-providers CUDAExecutionProvider` to run the check on a GPU.
