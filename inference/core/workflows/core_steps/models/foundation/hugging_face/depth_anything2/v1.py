"""
Credits to: https://github.com/Fafruch for origin idea
"""

from typing import List, Literal, Optional, Type

import numpy as np
from PIL import Image
from transformers import pipeline
import supervision as sv
from pydantic import ConfigDict, Field
from supervision import Color
import matplotlib

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SUPPORTED_MODEL_SIZES = ["Small", "Base", "Large"]
MODEL_SIZE_METADATA = {
    "Small": {
        "name": "Small Model",
        "description": "Lightweight model (25M parameters) with fastest inference time (~60ms). Best for resource-constrained environments.",
        "parameters": "25M",
        "latency": "60ms"
    },
    "Base": {
        "name": "Base Model",
        "description": "Medium-sized model (335M parameters) with balanced performance (~213ms). Suitable for most general applications.",
        "parameters": "335M",
        "latency": "213ms"
    },
    "Large": {
        "name": "Large Model",
        "description": "Large model (891M parameters) with highest accuracy but slower inference (~5.2s). Best for accuracy-critical applications.",
        "parameters": "891M",
        "latency": "5.2s"
    }
}

MODEL_SIZES_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in MODEL_SIZE_METADATA.items()
)

SUPPORTED_COLORMAPS = ["Spectral_r", "viridis", "plasma", "magma", "inferno"]
COLORMAP_METADATA = {
    "Spectral_r": {
        "name": "Spectral Reversed",
        "description": "Rainbow-like colormap that's effective for depth visualization, reversed for intuitive depth perception.",
    },
    "viridis": {
        "name": "Viridis",
        "description": "Perceptually uniform colormap that works well for colorblind viewers.",
    }, 
    "plasma": {
        "name": "Plasma",
        "description": "Sequential colormap with high perceptual contrast.",
    },
    "magma": {
        "name": "Magma",
        "description": "Sequential colormap with dark-to-light transition.",
    },
    "inferno": {
        "name": "Inferno",
        "description": "High-contrast sequential colormap with sharp visual distinction.",
    }
}

COLORMAP_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in COLORMAP_METADATA.items()
)

LONG_DESCRIPTION = """
Transform your 2D images into stunning depth maps with Depth Anything v2! 
This powerful tool helps you understand the 3D structure of any image by predicting how far each pixel is from the camera.

#### 🎯 How It Works

This block processes images by:

1. 📸 Taking your input image
2. 🤖 Running it through a state-of-the-art depth estimation model
3. 🎨 Creating beautiful depth visualizations using customizable colormaps
4. 📊 Providing normalized depth values for further processing

#### 🚀 Available Models

Choose the model that best fits your needs:

{MODEL_SIZES_DOCS_DESCRIPTION}

#### 🎨 Visualization Options

Make your depth maps pop with these colormap options:

{COLORMAP_DOCS_DESCRIPTION}

#### 💡 Why Use Depth Anything v2?

This block is perfect for:

- 🏗️ 3D reconstruction projects
- 🤖 Robotics applications needing depth perception
- 🔍 Scene understanding tasks
- 📏 Distance estimation applications

#### 🛠️ Output Format

The block provides two outputs:
1. A colored visualization of the depth map using your chosen colormap
2. A normalized depth array (0-1 range) for technical applications

#### 💪 Key Features

- 🎯 State-of-the-art depth estimation
- 🎨 Multiple colormap options for different visualization needs
- ⚡ Flexible model sizes for speed/accuracy tradeoffs
- 📊 Normalized depth values for technical applications
- 🔧 Easy integration with other workflow blocks

#### 🎯 Perfect For

- 👨‍💻 Developers working on 3D reconstruction
- 🎨 Artists creating depth-based effects
- 🤖 Robotics engineers building perception systems
- 📸 Photographers exploring depth visualization
"""

SHORT_DESCRIPTION = "Predicts depth maps from images"

ModelSize = Literal[tuple(SUPPORTED_MODEL_SIZES)] # type: ignore
ColormapType = Literal[tuple(SUPPORTED_COLORMAPS)]  # type: ignore


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Depth Anything v2",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Huggingface",
                "huggingface",
                "depth anything v2",
                "depth prediction",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-palette",
            },
            "task_type_property": "model_size",
        }
    )
    type: Literal["roboflow_core/depth_anything_v2@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The image from which to predict depth",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    model_size: ModelSize = Field(
        default="base",
        description="Size of the model to use for depth prediction",
        json_schema_extra={
            "values_metadata": MODEL_SIZE_METADATA,
            "always_visible": True,
        },
    )
    colormap: ColormapType = Field(
        default="Spectral_r",
        description="Colormap to use for depth visualization",
        json_schema_extra={
            "values_metadata": COLORMAP_METADATA,
            "always_visible": True,
        },
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
            OutputDefinition(name="normalized_depth", kind=[NUMPY_ARRAY_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"




class DepthAnythingV2BlockV1(WorkflowBlock):
    def __init__(self):
        super().__init__()
        self._pipe = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        model_size: ModelSize,
        colormap: ColormapType,
    ) -> BlockResult:
        # Convert input image
        numpy_image = image.numpy_image
        pil_image = Image.fromarray(numpy_image)

        try:
            # Initialize or get cached pipeline
            if self._pipe is None:
                self._pipe = get_depth_pipeline(model_size)
            
            # Get depth prediction
            depth = np.array(self._pipe(pil_image)["depth"])
            
            # Process depth map
            depth = process_depth_map(depth)
            colored_depth = create_visualization(depth, colormap)
            normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())

            return {
                'image': WorkflowImageData.copy_and_replace(
                    origin_image_data=image,
                    numpy_image=colored_depth,
                ),
                'normalized_depth': normalized_depth
            }
        except Exception as e:
            raise RuntimeError(f"Failed to process depth estimation: {str(e)}")


def get_depth_pipeline(model_size: ModelSize):
    """Initialize depth estimation pipeline."""
    return pipeline(
        task="depth-estimation", 
        model=f"depth-anything/Depth-Anything-V2-{model_size}-hf"
    )

def process_depth_map(depth_array: np.ndarray) -> np.ndarray:
    """Process and validate depth map."""
    if depth_array.max() == depth_array.min():
        raise ValueError("Depth map has no variation (min equals max)")
    return depth_array

def create_visualization(depth_array: np.ndarray, colormap: ColormapType) -> np.ndarray:
    """Create colored visualization of depth map."""
    # Normalize depth for visualization based on its own min and max
    depth_min, depth_max = depth_array.min(), depth_array.max()
    depth_for_viz = ((depth_array - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
    
    cmap = matplotlib.colormaps.get_cmap(colormap)
    return (cmap(depth_for_viz)[:, :, :3] * 255).astype(np.uint8)