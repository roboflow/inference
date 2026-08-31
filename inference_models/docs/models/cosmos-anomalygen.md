# Cosmos AnomalyGen - Synthetic Defect Generation

Cosmos AnomalyGen (NVIDIA `paidf-anomalygen`) is a generative defect-inpainting model: given a
clean image, a binary placement mask, and a trained anomaly type, it inpaints a realistic defect
inside the mask. It is fine-tuned per dataset on a handful of labeled defect images
(instance-segmentation data) and then used to manufacture synthetic training images.

## Overview

- **Mask-Conditioned Inpainting** - defects appear only where the placement mask says
- **Few-Shot Fine-Tuning** - a trained checkpoint is a ~14 MB adapter + per-class anomaly
  embeddings on top of a frozen Cosmos-Predict2-2B-Text2Image base
- **SDG-Compatible Parameters** - `guidance`, `num_steps`, `seed`, `crop_and_paste`,
  `crop_ratio`, `poisson_blend` mirror NVIDIA's `synthetic_dataset_generation` JSONL contract
  one entry to one call

!!! info "License & Attribution"
    **License**: NVIDIA OSPA / Apache-2.0 components - licensing review pending<br>
    **Source**: [NVIDIA/paidf-anomalygen](https://github.com/NVIDIA/paidf-anomalygen)<br>
    **Base model**: [Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image)

!!! note "Runtime Requirements"
    The model runs through NVIDIA's GA stack (`cosmos_predict2` + `imaginaire`), which is **not**
    installable from PyPI. Serve it inside a container built on the `paidf-anomalygen:ga` base
    image with `inference` installed on top. The runtime module ships inside the weight package
    (`cosmos_anomalygen_runtime.py`), so loading it requires either a package marked as trusted or
    `ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES=True`. Needs a GPU with >=16 GB VRAM
    (~14 GB in use during generation, ~10 s/image on an L4).

## Model IDs

Checkpoints are fine-tuned per dataset; there is no useful zero-shot checkpoint. Each trained
model is registered as architecture `cosmos-anomalygen` (task `image-generation`, backend
`custom`), variant `cosmos-anomalygen-2b`.

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `custom` | none installable - requires the `paidf-anomalygen:ga` container environment |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Fine-tune on instance-segmentation projects (COCO polygon export) |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ❌ Not available |
| **Workflows** | ✅ `roboflow_core/cosmos_anomalygen@v1` block (local GPU execution only) |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ⚠️ GA-based container only (see Runtime Requirements) |

## Usage Examples

### Generate a synthetic defect

```python
import cv2
import numpy as np
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "path/to/cosmos-anomalygen-package",  # or a registered model id
    api_key="your_roboflow_api_key",
)

image = cv2.imread("clean_part.png")
mask = np.zeros(image.shape[:2], dtype=np.uint8)
mask[200:320, 180:300] = 255  # where the defect should appear

generated = model.generate(
    image=image,
    mask=mask,
    anomaly_type="tube+hole",  # a trained "<category>+<class>" pair
    guidance=1.5,
    num_steps=35,
    seed=0,
)
cv2.imwrite("synthetic_defect.png", generated[0])
```

### Filtering empty generations

The model sometimes returns the canvas unchanged (more often at low guidance). Measure
visibility - the mean absolute gray-level change inside the mask - and regenerate with a new
seed when it is low; `>=15` separated real defects from empty generations in practice:

```python
gray_before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_after = cv2.cvtColor(generated[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
visibility = np.abs(gray_after - gray_before)[mask >= 128].mean()
```

The workflow block reports this as its `visibility` output.

### Practical notes

- `guidance` is an extrapolation scale (`x0 = cond + g*(cond - uncond)`); the production default
  `1.5` corresponds to a standard classifier-free-guidance scale of `2.5`.
- `crop_ratio` sizes the 512 px generation window relative to the mask's bounding box. With the
  default `4.0`, defects larger than ~1/4 of the frame make the crop exceed the frame and the
  whole image gets resampled through the VAE - lower the ratio for large defects.
- The generated defect can bleed up to one 8 px latent cell past the input mask. If you
  auto-annotate generated images, derive labels from the changed region, not the input polygon.
