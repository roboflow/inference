# Flex.2 Inpainting

Use segmentation masks to inpaint objects within an image using the Flex.2-preview model.

## Overview

The Flex.2 Inpainting block enables you to modify specific regions of an image using the state-of-the-art Flex.2-preview diffusion model. This block takes an input image, segmentation masks defining regions to modify, and text prompts describing the desired changes.

Flex.2 is currently the most flexible text-to-image diffusion model released, featuring:
- 8 billion parameters
- Built-in inpainting support
- Universal control input (line, pose, depth)
- 512 token length input
- 16 channel latent space
- Apache 2.0 license

## How It Works

1. **Input Processing**: The block receives an image and segmentation masks (from a segmentation model)
2. **Mask Creation**: Segmentation predictions are converted into inpainting masks
3. **Model Inference**: The Flex.2 model generates new content for masked regions based on your prompt
4. **Output**: Returns the modified image with inpainted regions

## Installation

This block requires the transformers dependencies to be installed:

```bash
pip install inference[transformers]
```

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | Image | - | The image to inpaint |
| segmentation_mask | Segmentation | - | Segmentation predictions defining regions to inpaint |
| prompt | String | - | Text describing what you want to see in the inpainted regions |
| negative_prompt | String | None | Text describing what you don't want to see |
| control_strength | Float | 0.5 | Strength of control input (0.0-1.0) |
| control_stop | Float | 0.33 | When to stop applying control during generation (0.0-1.0) |
| num_inference_steps | Integer | 50 | Number of denoising steps |
| guidance_scale | Float | 3.5 | How closely to follow the prompt (higher = more adherence) |
| seed | Integer | None | Random seed for reproducible results |
| height | Integer | 1024 | Output image height (must be divisible by 16) |
| width | Integer | 1024 | Output image width (must be divisible by 16) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| image | Image | The inpainted image |

## Example Usage

### Basic Inpainting

Replace detected objects with new content:

```python
# Detect objects
segmentation = segment_anything_2(image=input_image)
# Inpaint the detected regions
result = flex2_inpainting(
    image=input_image,
    segmentation_mask=segmentation,
    prompt="A beautiful garden with flowers",
    negative_prompt="people, cars, buildings"
)
```

### Advanced Parameters

Fine-tune the generation process:

```python
result = flex2_inpainting(
    image=input_image,
    segmentation_mask=segmentation,
    prompt="A serene lake surrounded by mountains",
    negative_prompt="urban, city, pollution",
    num_inference_steps=100,  # More steps for higher quality
    guidance_scale=7.5,       # Stronger prompt adherence
    seed=42,                  # Reproducible results
    height=1024,
    width=1024
)
```

## Tips and Best Practices

1. **Prompt Engineering**: Be specific and descriptive in your prompts for best results
2. **Negative Prompts**: Use negative prompts to exclude unwanted elements
3. **Inference Steps**: More steps (50-100) generally produce higher quality but take longer
4. **Guidance Scale**: Values between 3-10 work well; higher values follow prompts more strictly5. **Seed**: Set a seed value for reproducible results across runs
6. **Resolution**: Use dimensions divisible by 16 for best compatibility

## GPU Requirements

- Recommended: GPU with at least 16GB VRAM
- The model will automatically use CUDA if available, otherwise CPU (slower)
- For lower VRAM GPUs, consider using smaller image dimensions

## Model Information

This block uses the [Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) model by ostris, which is built on top of the Flux architecture with significant improvements for flexibility and control.

## Credits

Model developed by ostris and the open-source community. Licensed under Apache 2.0.
