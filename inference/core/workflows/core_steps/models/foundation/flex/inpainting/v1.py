"""
Flex.2-preview Inpainting - local execution only using HuggingFace's implementation
"""

from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode

# Global cache for the Flex.2 pipeline
_CACHED_FLEX2_PIPELINE = None

LONG_DESCRIPTION = """
Use segmentation masks to inpaint objects within an image using Flex.2-preview model.

Flex.2 is the most flexible text-to-image diffusion model released, with built-in inpainting 
and universal control support. This block runs the model locally using HuggingFace's diffusers library.

Model features:
- 8 billion parameters
- Built-in inpainting support
- Universal control input (line, pose, depth)
- 512 token length input
- 16 channel latent space

Based on [Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) by ostris.
"""

SHORT_DESCRIPTION = "Use segmentation masks to inpaint objects using Flex.2-preview model."

class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Flex.2 Inpainting",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Flex.2",
                "Flex2",
                "inpainting",
                "image generation",
                "diffusion",
                "ostris",
                "huggingface",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fas fa-paint-brush",
                "blockPriority": 15,
            },
        }
    )
    type: Literal["roboflow_core/flex2_inpainting@v1"]
    
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The image to inpaint.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )    
    segmentation_mask: Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]) = Field(
        name="Segmentation Mask",
        description="Model predictions from segmentation model.",
        examples=["$steps.model.predictions"],
    )
    
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Prompt to inpainting model (what you wish to see).",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )
    
    negative_prompt: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default=None,
        description="Negative prompt to inpainting model (what you do not wish to see).",
        examples=["my negative prompt", "$inputs.negative_prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )    
    # Control parameters
    control_strength: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=0.5,
        description="Strength of the control input (0.0 to 1.0).",
        examples=[0.5, "$inputs.control_strength"],
    )
    
    control_stop: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=0.33,
        description="When to stop applying control during generation (0.0 to 1.0).",
        examples=[0.33, "$inputs.control_stop"],
    )
    
    # Generation parameters
    num_inference_steps: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=50,
        description="Number of denoising steps.",
        examples=[50, "$inputs.num_inference_steps"],
    )
    
    guidance_scale: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=3.5,
        description="Guidance scale. Higher values produce images more closely linked to the prompt.",
        examples=[3.5, "$inputs.guidance_scale"],
    )
    
    seed: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=None,
        description="Random seed for reproducible generation.",
        examples=[42, "$inputs.seed"],
    )
    height: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=1024,
        description="Height of the generated image. Should be divisible by 16.",
        examples=[1024, "$inputs.height"],
    )
    
    width: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=1024,
        description="Width of the generated image. Should be divisible by 16.",
        examples=[1024, "$inputs.width"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class Flex2InpaintingBlockV1(WorkflowBlock):
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        api_key: Optional[str] = None,
        step_execution_mode: Optional[StepExecutionMode] = None,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        segmentation_mask: sv.Detections,
        prompt: str,
        negative_prompt: Optional[str],
        control_strength: float,
        control_stop: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        height: int,
        width: int,
    ) -> BlockResult:
        # Create mask from segmentation
        full_mask = np.zeros((image.numpy_image.shape[0], image.numpy_image.shape[1]), dtype=np.uint8)
        
        if segmentation_mask.mask is not None and len(segmentation_mask.mask) > 0:
            # Combine all masks
            for single_mask in segmentation_mask.mask:
                full_mask = np.logical_or(full_mask, single_mask).astype(np.uint8) * 255
        else:            # Fallback to bounding boxes if no masks
            for xyxy in segmentation_mask.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                full_mask[y1:y2, x1:x2] = 255
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(full_mask, (15, 15), 0)
        
        result_image = self._run_inference(
            image=image.numpy_image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            control_stop=control_stop,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            height=height,
            width=width,
        )
        
        return {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=result_image,
            ),
        }

    def _run_inference(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: Optional[str],
        control_strength: float,
        control_stop: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Run inference using local Flex.2 model."""
        try:
            from diffusers import AutoPipelineForText2Image
            from diffusers.utils import load_image
            import torch
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "Local execution requires 'diffusers' and 'torch'. "
                "Please install with: pip install inference[transformers]"
            )
        
        # Model caching is handled at module level to avoid reloading
        global _CACHED_FLEX2_PIPELINE
        
        device = "cuda" if torch.cuda.is_available() else "cpu"        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        if '_CACHED_FLEX2_PIPELINE' not in globals() or _CACHED_FLEX2_PIPELINE is None:
            logger.info(f"Loading Flex.2-preview model on {device}...")
            
            model_id = "ostris/Flex.2-preview"
            
            try:
                _CACHED_FLEX2_PIPELINE = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    custom_pipeline=model_id,
                    torch_dtype=dtype,
                )
                
                _CACHED_FLEX2_PIPELINE = _CACHED_FLEX2_PIPELINE.to(device)
                
                logger.info("Flex.2 model loaded successfully!")
                
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Flex.2 pipeline: {str(e)}")
        
        pipe = _CACHED_FLEX2_PIPELINE
        
        # Convert images to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Resize to target dimensions
        pil_image = pil_image.resize((width, height), PILImage.LANCZOS)
        
        # Convert mask to single channel if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)        # Resize mask to match target dimensions
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        pil_mask = PILImage.fromarray(mask_resized)
        
        # For Flex.2, we don't need a control image for basic inpainting
        # Set control_image to None to indicate no control input
        control_image = None
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Run inference
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                inpaint_image=pil_image,
                inpaint_mask=pil_mask,
                control_image=control_image,
                control_strength=control_strength,
                control_stop=control_stop,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
        
        # Convert back to OpenCV format
        result_array = np.array(result)        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
