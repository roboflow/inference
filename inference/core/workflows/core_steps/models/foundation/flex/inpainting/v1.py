"""
Flex.2-preview Inpainting - local execution only using HuggingFace's implementation
"""

from typing import List, Literal, Optional, Type, Union
import sys
import os
from pathlib import Path

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

**IMPORTANT: Memory Requirements**
- GPU: 16-24GB VRAM recommended
- CPU: 40-80GB RAM required (not recommended)
- The model files are ~30GB on disk

For production use, a GPU with sufficient VRAM is strongly recommended.

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
    def _load_custom_pipeline(self, model_id: str, dtype):
        """Manually load the Flex2Pipeline from the cached model files."""
        try:
            from diffusers import FluxControlPipeline
            from transformers import T5EncoderModel, CLIPTextModel
            from diffusers import FluxTransformer2DModel, AutoencoderKL
            import torch
            
            # First, check if the pipeline.py exists in the cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_cache_name = f"models--{model_id.replace('/', '--')}"
            model_cache_path = cache_dir / model_cache_name
            
            if not model_cache_path.exists():
                raise RuntimeError(f"Model cache not found at {model_cache_path}")
            
            # Find the snapshot directory
            snapshots_dir = model_cache_path / "snapshots"
            if not snapshots_dir.exists():
                raise RuntimeError(f"Snapshots directory not found at {snapshots_dir}")
            
            # Get the latest snapshot
            snapshot_dirs = list(snapshots_dir.iterdir())
            if not snapshot_dirs:
                raise RuntimeError("No snapshots found in cache")
            
            snapshot_dir = snapshot_dirs[0]  # Use first snapshot
            pipeline_path = snapshot_dir / "pipeline.py"
            
            if not pipeline_path.exists():
                raise RuntimeError(f"pipeline.py not found at {pipeline_path}")
            
            logger.info(f"Found custom pipeline at {pipeline_path}")
            
            # Dynamically import the pipeline
            import importlib.util
            spec = importlib.util.spec_from_file_location("flex2_pipeline", pipeline_path)
            pipeline_module = importlib.util.module_from_spec(spec)
            sys.modules["flex2_pipeline"] = pipeline_module
            spec.loader.exec_module(pipeline_module)
            
            # Get the Flex2Pipeline class
            Flex2Pipeline = pipeline_module.Flex2Pipeline
            
            # Load components individually
            logger.info("Loading model components...")
            
            scheduler = None
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
            text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
            tokenizer = None  # Will be loaded by from_pretrained
            text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
            tokenizer_2 = None  # Will be loaded by from_pretrained
            transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
            
            # Create the pipeline using from_pretrained to get all components properly initialized
            pipeline = Flex2Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            
            logger.info("Custom Flex2Pipeline loaded successfully!")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to manually load custom pipeline: {str(e)}")
            raise
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
        
        # For CPU, we need to be more careful about memory
        if device == "cpu":
            logger.warning(
                "Running Flex.2 on CPU requires significant memory (40-80GB). "
                "Consider using a GPU or a smaller model."
            )
            # Keep float32 on CPU - float16 is not supported on CPU
            dtype = torch.float32
        
        if '_CACHED_FLEX2_PIPELINE' not in globals() or _CACHED_FLEX2_PIPELINE is None:
            logger.info(f"Loading Flex.2-preview model on {device}...")
            
            model_id = "ostris/Flex.2-preview"
            
            try:
                # Try manual loading first
                logger.info("Attempting manual loading of custom pipeline...")
                _CACHED_FLEX2_PIPELINE = self._load_custom_pipeline(model_id, dtype)
            except Exception as e:
                logger.warning(f"Manual loading failed: {str(e)}")
                
                # Fallback to standard loading with memory optimizations
                try:
                    logger.info("Attempting standard loading with memory optimizations...")
                    
                    # For CPU, use sequential loading to reduce peak memory
                    if device == "cpu":
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                        
                    _CACHED_FLEX2_PIPELINE = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,  # This helps reduce memory usage during loading
                        device_map="auto" if device == "cuda" else None,  # Auto device mapping for GPU
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load Flex.2 model.\n"
                        f"Manual loading error: {str(e)}\n"
                        f"Standard loading error: {str(e2)}\n\n"
                        f"Please ensure:\n"
                        f"1. You have the latest diffusers version\n"
                        f"2. The model is fully downloaded\n"
                        f"3. You have sufficient disk space\n\n"
                        f"For CPU execution, Flex.2 requires 40-80GB of RAM.\n"
                        f"Consider using a GPU or a smaller model for CPU inference."
                    )                
            _CACHED_FLEX2_PIPELINE = _CACHED_FLEX2_PIPELINE.to(device)
            
            # Enable memory optimizations if available
            if device == "cpu":
                # For CPU, ensure all components are on CPU
                for component_name, component in _CACHED_FLEX2_PIPELINE.components.items():
                    if hasattr(component, "to"):
                        _CACHED_FLEX2_PIPELINE.components[component_name] = component.to(device)
            elif hasattr(_CACHED_FLEX2_PIPELINE, "enable_model_cpu_offload"):
                _CACHED_FLEX2_PIPELINE.enable_model_cpu_offload()
            elif hasattr(_CACHED_FLEX2_PIPELINE, "enable_attention_slicing"):
                _CACHED_FLEX2_PIPELINE.enable_attention_slicing()
                
            logger.info(f"Flex.2 model loaded successfully! Pipeline type: {type(_CACHED_FLEX2_PIPELINE).__name__}")
        
        pipe = _CACHED_FLEX2_PIPELINE
        
        # Ensure dimensions are compatible with the model
        if height % 16 != 0 or width % 16 != 0:
            logger.warning(f"Dimensions {width}x{height} not divisible by 16, adjusting...")
            height = (height // 16) * 16
            width = (width // 16) * 16
            logger.info(f"Adjusted dimensions to {width}x{height}")
        
        # Convert images to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((width, height), PILImage.LANCZOS)
        
        # Convert mask to single channel if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # Resize mask to match target dimensions
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        pil_mask = PILImage.fromarray(mask_resized)
        
        # For Flex.2, we don't need a control image for basic inpainting
        control_image = None
        
        # Check the pipeline type
        pipeline_class_name = type(pipe).__name__
        
        # Workaround for bug in Flex2Pipeline where it tries to access control_image.shape
        if "Flex2Pipeline" in pipeline_class_name:
            # Create a zero-filled control image as a workaround
            control_image = PILImage.new('RGB', (width, height), color=(0, 0, 0))
            logger.info("Created placeholder control image to work around pipeline bug")
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Run inference
        with torch.inference_mode():
            logger.info(f"Running inference with pipeline: {pipeline_class_name}")
            
            try:
                # Check if this is Flex2Pipeline
                if hasattr(pipe, '__call__'):
                    # Use the parameters from the Flex2Pipeline
                    result = pipe(
                        prompt=prompt,
                        prompt_2=negative_prompt,  # Flex2 uses prompt_2 for negative
                        inpaint_image=pil_image,
                        inpaint_mask=pil_mask,
                        control_image=control_image,
                        control_strength=control_strength,
                        control_stop=control_stop,
                        height=height,                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    ).images[0]
                    logger.info("Successfully generated image with inpainting")
                else:
                    raise RuntimeError(f"Pipeline {pipeline_class_name} is not callable")
                    
            except Exception as e:
                logger.error(f"Failed to run inference: {str(e)}")
                
                # Check if it's the FluxPipeline without inpainting support
                if "FluxPipeline" in pipeline_class_name and "inpaint" in str(e):
                    raise RuntimeError(
                        f"The Flex.2 custom pipeline failed to load. Got {pipeline_class_name} instead.\n"
                        f"This block requires the custom Flex2Pipeline for inpainting support.\n\n"
                        f"Error: {str(e)}"
                    )
                else:
                    raise
        
        # Convert back to OpenCV format
        result_array = np.array(result)
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)