# Stability AI Inpainting

<a href="https://stability.ai/" target="_blank">Stability AI</a> provides state-of-the-art image generation models including Stable Diffusion for inpainting tasks.

The Stability AI Inpainting block in Inference allows you to use segmentation masks to intelligently replace or modify specific regions of an image using text prompts.

## Execution Modes

The block supports two execution modes:

### Cloud Mode (Default)
- Uses Stability AI's cloud API
- Requires a Stability AI API key
- Faster for single images
- No local GPU required

### Local Mode
- Runs Stable Diffusion Inpainting v1.5 locally
- Requires transformers dependencies
- Better for batch processing
- Requires GPU for optimal performance

## Installation

To use the Stability AI Inpainting block with local execution, install Inference with transformers support:

```bash
pip install inference[transformers]
```

or for GPU support:

```bash
pip install inference-gpu[transformers]
```

For cloud execution only, the standard Inference installation is sufficient.

## How to Use Stability AI Inpainting

### Cloud Execution Example

```python
import cv2
import supervision as sv
from inference import get_model
from inference.core.workflows.core_steps.models.foundation.stability_ai.inpainting.v2 import (
    StabilityAIInpaintingBlockV2
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# Load your image
image = cv2.imread("path/to/image.jpg")
workflow_image = WorkflowImageData(numpy_image=image)

# Get segmentation mask (example using SAM)
sam = get_model("sam/vit_h")
masks = sam.infer(image)

# Convert to supervision detections format
detections = sv.Detections(
    xyxy=masks["xyxy"],
    mask=masks["mask"],
)

# Run inpainting
block = StabilityAIInpaintingBlockV2()
result = block.run(
    image=workflow_image,
    segmentation_mask=detections,
    prompt="beautiful flowers",
    negative_prompt="blurry, low quality",
    execution_mode="cloud",
    api_key="your-stability-api-key",
    num_inference_steps=None,  # Not used in cloud mode
    guidance_scale=None,       # Not used in cloud mode
    seed=None,                 # Not used in cloud mode
)

# Save result
cv2.imwrite("inpainted.jpg", result["image"].numpy_image)
```

### Local Execution Example

```python
# Same setup as above...

# Run inpainting locally
result = block.run(
    image=workflow_image,
    segmentation_mask=detections,
    prompt="beautiful flowers",
    negative_prompt="blurry, low quality",
    execution_mode="local",
    api_key=None,  # Not needed for local execution
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,  # For reproducible results
)
```

## Workflow Usage

The block can be used in Inference Workflows:

```yaml
blocks:
  - type: roboflow_core/roboflow_instance_segmentation_model@v2
    name: segmentation
    model_id: your-segmentation-model
    
  - type: roboflow_core/stability_ai_inpainting@v2
    name: inpainting
    execution_mode: cloud
    image: $inputs.image
    segmentation_mask: $segmentation.predictions
    prompt: $inputs.inpaint_prompt
    api_key: $inputs.stability_api_key
```

## Parameters

### Common Parameters
- `image`: The input image to inpaint
- `segmentation_mask`: Instance segmentation predictions defining areas to inpaint
- `prompt`: Text description of what to generate in the masked area
- `negative_prompt`: (Optional) Text description of what to avoid
- `execution_mode`: Either "cloud" or "local"

### Cloud-Specific Parameters
- `api_key`: Your Stability AI API key (required)

### Local-Specific Parameters
- `num_inference_steps`: Number of denoising steps (default: 50)
- `guidance_scale`: How closely to follow the prompt (default: 7.5)
- `seed`: Random seed for reproducible generation (optional)

## License

When using local execution mode, the model runs under the <a href="https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/LICENSE" target="_blank">CreativeML Open RAIL-M</a> license.

Cloud execution is subject to Stability AI's terms of service.
