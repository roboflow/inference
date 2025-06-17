"""
Stability AI Inpainting v2 - supports both cloud API and local execution
"""

import os
from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import requests
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
    SECRET_KIND,
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

# Global cache for the Stable Diffusion pipeline
_CACHED_SD_PIPELINE = None

LONG_DESCRIPTION = """
Use segmentation masks to inpaint objects within an image using Stable Diffusion.

This block supports two execution modes:
- **Cloud**: Uses Stability AI's cloud API (requires API key)
- **Local**: Runs Stable Diffusion Inpainting v1.5 locally (requires transformers)

The block wraps [Stability AI inpainting API](https://platform.stability.ai/docs/legacy/grpc-api/features/inpainting#Python) 
for cloud execution and uses HuggingFace's diffusers library for local execution.
"""

SHORT_DESCRIPTION = "Use segmentation masks to inpaint objects within an image."

API_HOST = "https://api.stability.ai"
ENDPOINT = "/v2beta/stable-image/edit/inpaint"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stability AI Inpainting",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Stability AI",
                "stability.ai",
                "inpainting",
                "image generation",
                "stable diffusion",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-palette",
                "blockPriority": 14,
            },
        }
    )
    type: Literal["roboflow_core/stability_ai_inpainting@v2"]
    
    execution_mode: Union[
        Literal["cloud", "local"],
        Selector(kind=[STRING_KIND])
    ] = Field(
        default="cloud",
        description="Execution mode - 'cloud' uses Stability AI API, 'local' runs locally",
        examples=["cloud", "local"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    
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
    
    # Cloud-specific parameters
    api_key: Optional[Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str]] = Field(
        default=None,
        description="Your Stability AI API key (required for cloud execution).",
        examples=["xxx-xxx", "$inputs.stability_ai_api_key"],
        private=True,
        json_schema_extra={
            "relevant_for": {
                "execution_mode": {"values": ["cloud"], "required": True},
            },
        },
    )
    
    # Local-specific parameters
    num_inference_steps: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=50,
        description="Number of denoising steps for local execution.",
        examples=[50, "$inputs.num_inference_steps"],
        json_schema_extra={
            "relevant_for": {
                "execution_mode": {"values": ["local"], "required": False},
            },
        },
    )
    
    guidance_scale: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=7.5,
        description="Guidance scale for local execution. Higher values produce images more closely linked to the prompt.",
        examples=[7.5, "$inputs.guidance_scale"],
        json_schema_extra={
            "relevant_for": {
                "execution_mode": {"values": ["local"], "required": False},
            },
        },
    )
    
    seed: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        default=None,
        description="Random seed for reproducible generation in local execution.",
        examples=[42, "$inputs.seed"],
        json_schema_extra={
            "relevant_for": {
                "execution_mode": {"values": ["local"], "required": False},
            },
        },
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class StabilityAIInpaintingBlockV2(WorkflowBlock):
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
        execution_mode: str,
        api_key: Optional[str],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        seed: Optional[int],
    ) -> BlockResult:
        # Validate API key for cloud mode
        if execution_mode == "cloud" and not api_key:
            raise ValueError("API key is required when execution_mode is 'cloud'")
        
        # Create mask from segmentation
        black_image = np.zeros_like(image.numpy_image)
        # Create a simple mask without using MaskAnnotator to avoid class_id issues
        full_mask = np.zeros((image.numpy_image.shape[0], image.numpy_image.shape[1]), dtype=np.uint8)
        
        if segmentation_mask.mask is not None and len(segmentation_mask.mask) > 0:
            # Combine all masks
            for single_mask in segmentation_mask.mask:
                full_mask = np.logical_or(full_mask, single_mask).astype(np.uint8) * 255
        else:
            # Fallback to bounding boxes if no masks
            for xyxy in segmentation_mask.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                full_mask[y1:y2, x1:x2] = 255
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(full_mask, (15, 15), 0)
        
        if execution_mode == "cloud":
            result_image = self._run_cloud_inference(
                image=image.numpy_image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                api_key=api_key,
            )
        else:  # local
            result_image = self._run_local_inference(
                image=image.numpy_image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        
        return {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=result_image,
            ),
        }

    def _run_cloud_inference(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: Optional[str],
        api_key: str,
    ) -> np.ndarray:
        """Run inference using Stability AI cloud API."""
        encoded_image = numpy_array_to_jpeg_bytes(image=image)
        encoded_mask = numpy_array_to_jpeg_bytes(image=mask)
        
        request_data = {
            "prompt": prompt,
            "output_format": "jpeg",
        }
        
        if negative_prompt:
            request_data["negative_prompt"] = negative_prompt
        
        response = requests.post(
            f"{API_HOST}{ENDPOINT}",
            headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
            files={
                "image": encoded_image,
                "mask": encoded_mask,
            },
            data=request_data,
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Request to StabilityAI API failed: {str(response.json())}"
            )
        
        return bytes_to_opencv_image(payload=response.content)

    def _run_local_inference(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> np.ndarray:
        """Run inference using local Stable Diffusion model."""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "Local execution requires 'diffusers' and 'torch'. "
                "Please install with: pip install inference[transformers]"
            )
        
        # Model caching is handled at module level to avoid reloading
        global _CACHED_SD_PIPELINE
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if '_CACHED_SD_PIPELINE' not in globals() or _CACHED_SD_PIPELINE is None:
            logger.info(f"Loading Stable Diffusion Inpainting model on {device}...")
            
            # Use the correct model ID with actual model files
            model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"
            
            try:
                # First attempt with local files only to check if already downloaded
                try:
                    _CACHED_SD_PIPELINE = StableDiffusionInpaintPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        variant="fp16" if device == "cuda" else None,  # Use fp16 variant
                        use_safetensors=True,
                        local_files_only=True,  # Try local first
                    )
                    logger.info("Loaded model from local cache")
                except Exception as local_error:
                    # If local loading fails, allow downloading
                    logger.info("Model not found locally, downloading from HuggingFace...")
                    _CACHED_SD_PIPELINE = StableDiffusionInpaintPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        variant="fp16" if device == "cuda" else None,  # Use fp16 variant
                        use_safetensors=True,
                        local_files_only=False,  # Allow downloading
                        resume_download=True,  # Resume if partially downloaded
                    )
                    logger.info("Model downloaded successfully")
                
                _CACHED_SD_PIPELINE = _CACHED_SD_PIPELINE.to(device)
                
                # Enable memory optimizations
                if hasattr(_CACHED_SD_PIPELINE, "enable_attention_slicing"):
                    _CACHED_SD_PIPELINE.enable_attention_slicing()
                
                logger.info("Stable Diffusion model loaded successfully!")
                
            except Exception as e:
                error_msg = str(e).lower()
                if "no file named" in error_msg or "not found" in error_msg:
                    raise RuntimeError(
                        f"Failed to load Stable Diffusion model. The model files may not be "
                        f"fully downloaded. This model requires approximately 4-5GB of disk space.\n\n"
                        f"To download the model, run this Python code:\n"
                        f"from diffusers import StableDiffusionInpaintPipeline\n"
                        f"pipe = StableDiffusionInpaintPipeline.from_pretrained('{model_id}')\n\n"
                        f"Original error: {str(e)}"
                    )
                else:
                    raise RuntimeError(f"Failed to initialize Stable Diffusion pipeline: {str(e)}")
        
        pipe = _CACHED_SD_PIPELINE
        
        # Convert images to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Convert mask to single channel if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pil_mask = PILImage.fromarray(mask)
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Run inference
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        
        # Convert back to OpenCV format
        result_array = np.array(result)
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)



def numpy_array_to_jpeg_bytes(
    image: np.ndarray,
) -> bytes:
    _, img_encoded = cv2.imencode(".jpg", image)
    return np.array(img_encoded).tobytes()


def bytes_to_opencv_image(
    payload: bytes, array_type: np.number = np.uint8
) -> np.ndarray:
    bytes_array = np.frombuffer(payload, dtype=array_type)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    if decoding_result is None:
        raise ValueError("Could not encode bytes to OpenCV image.")
    return decoding_result
