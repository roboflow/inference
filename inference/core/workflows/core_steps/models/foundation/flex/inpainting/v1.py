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
            from diffusers import AutoPipelineForText2Image, DiffusionPipeline
            import torch
            from PIL import Image as PILImage
            import os
            from pathlib import Path
        except ImportError:
            raise ImportError(
                "Local execution requires 'diffusers' and 'torch'. "
                "Please install with: pip install inference[transformers]"
            )
        
        # Model caching is handled at module level to avoid reloading
        global _CACHED_FLEX2_PIPELINE
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        if '_CACHED_FLEX2_PIPELINE' not in globals() or _CACHED_FLEX2_PIPELINE is None:
            logger.info(f"Loading Flex.2-preview model on {device}...")
            
            model_id = "ostris/Flex.2-preview"
            
            try:
                # First attempt: Try loading with AutoPipelineForText2Image
                logger.info("Attempting to load Flex.2 with AutoPipelineForText2Image...")
                _CACHED_FLEX2_PIPELINE = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    custom_pipeline=model_id,  # This loads the custom pipeline.py from the repo
                    torch_dtype=dtype,
                    trust_remote_code=True,  # Allow custom pipeline code
                    use_safetensors=True,
                )
            except Exception as e:
                logger.warning(f"AutoPipelineForText2Image failed: {str(e)}")
                logger.info("Attempting alternative loading method with DiffusionPipeline...")
                
                try:
                    # Second attempt: Try DiffusionPipeline which might handle custom pipelines better
                    _CACHED_FLEX2_PIPELINE = DiffusionPipeline.from_pretrained(
                        model_id,
                        custom_pipeline=model_id,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                except Exception as e2:
                    logger.warning(f"DiffusionPipeline with custom_pipeline failed: {str(e2)}")
                    
                    # Final attempt: Load without custom_pipeline and let it figure out the pipeline
                    logger.info("Attempting to load without custom_pipeline parameter...")
                    _CACHED_FLEX2_PIPELINE = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                
                _CACHED_FLEX2_PIPELINE = _CACHED_FLEX2_PIPELINE.to(device)
                
                # Enable memory optimizations if available
                if device == "cpu":
                    # For CPU, we need to ensure all components are on CPU
                    for component_name, component in _CACHED_FLEX2_PIPELINE.components.items():
                        if hasattr(component, "to"):
                            _CACHED_FLEX2_PIPELINE.components[component_name] = component.to(device)
                elif hasattr(_CACHED_FLEX2_PIPELINE, "enable_model_cpu_offload"):
                    _CACHED_FLEX2_PIPELINE.enable_model_cpu_offload()
                elif hasattr(_CACHED_FLEX2_PIPELINE, "enable_attention_slicing"):
                    _CACHED_FLEX2_PIPELINE.enable_attention_slicing()
                
                logger.info(f"Flex.2 model loaded successfully! Pipeline type: {type(_CACHED_FLEX2_PIPELINE).__name__}")
                
                # Check if this is actually the custom pipeline
                if hasattr(_CACHED_FLEX2_PIPELINE, '__call__'):
                    # Check for expected method signature
                    import inspect
                    sig = inspect.signature(_CACHED_FLEX2_PIPELINE.__call__)
                    params = list(sig.parameters.keys())
                    
                    if 'inpaint_image' in params and 'control_image' in params:
                        logger.info("✓ Custom Flex2Pipeline loaded successfully with inpainting support")
                    else:
                        logger.warning(
                            f"Pipeline loaded but may not support full Flex.2 features. "
                            f"Available parameters: {params[:10]}..."  # Show first 10 params
                        )
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Flex.2 pipeline: {str(e)}\n\n"
                    f"Please ensure you have the latest version of diffusers installed:\n"
                    f"pip install --upgrade diffusers transformers accelerate\n\n"
                    f"If the error persists, the model may still be downloading. "
                    f"Check your HuggingFace cache directory."
                )
        
        pipe = _CACHED_FLEX2_PIPELINE
        
        # Ensure dimensions are compatible with the model
        # Flux/Flex models need specific resolutions
        if height % 16 != 0 or width % 16 != 0:
            logger.warning(f"Dimensions {width}x{height} not divisible by 16, adjusting...")
            height = (height // 16) * 16
            width = (width // 16) * 16
            logger.info(f"Adjusted dimensions to {width}x{height}")
        
        # Convert images to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Resize to target dimensions
        pil_image = pil_image.resize((width, height), PILImage.LANCZOS)
        
        # Convert mask to single channel if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # Resize mask to match target dimensions
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
            pipeline_class_name = type(pipe).__name__
            logger.info(f"Running inference with pipeline: {pipeline_class_name}")
            
            # Based on the pipeline.py, Flex2Pipeline expects these parameters
            try:
                # Check if this is a Flex2Pipeline or similar
                if hasattr(pipe, 'inpaint_image') or 'Flex' in pipeline_class_name:
                    logger.info("Using Flex2Pipeline parameters")
                    result = pipe(
                        prompt=prompt,
                        prompt_2=prompt if negative_prompt is None else negative_prompt,  # Use negative_prompt as prompt_2
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
                else:
                    # For FluxPipeline or other pipelines
                    logger.info(f"Using standard pipeline parameters for {pipeline_class_name}")
                    # FluxPipeline doesn't support inpainting directly
                    # We'll need to implement inpainting logic ourselves or fail
                    raise RuntimeError(
                        f"Pipeline {pipeline_class_name} does not support inpainting. "
                        f"The Flex.2 custom pipeline may not have loaded correctly. "
                        f"Please ensure you have the latest version of diffusers installed."
                    )
                    
                logger.info("Successfully generated image with inpainting")
            except Exception as e:
                logger.error(f"Failed to run inference: {str(e)}")
                
                # If it's a dimension mismatch, provide helpful info
                if "tensor a" in str(e) and "must match the size of tensor b" in str(e):
                    raise RuntimeError(
                        f"Dimension mismatch error: {str(e)}\n\n"
                        f"This typically happens when the model expects specific resolutions.\n"
                        f"Current dimensions: {width}x{height}\n"
                        f"Try using standard Flux dimensions like 1024x1024 or 512x512.\n"
                        f"Pipeline type: {pipeline_class_name}"
                    )
                else:
                    raise RuntimeError(
                        f"Pipeline execution failed: {str(e)}\n\n"
                        f"Pipeline type: {pipeline_class_name}\n"
                        f"This may be due to:\n"
                        f"1. The custom pipeline not loading correctly\n"
                        f"2. Incompatible parameters\n"
                        f"3. Missing dependencies"
                    )
        
        # Convert back to OpenCV format
        result_array = np.array(result)
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
