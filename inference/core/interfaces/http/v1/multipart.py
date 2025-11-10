"""
Multipart form data handling utilities for v1 API.

This module provides efficient handling of multipart/form-data requests with
image uploads, avoiding the base64 encoding bottleneck of the legacy API.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import UploadFile
from starlette.datastructures import FormData

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.exceptions import InputImageLoadError
from inference.core import logger


async def parse_multipart_with_images(
    form_data: FormData,
    config_field: str = "config",
    image_fields: Optional[List[str]] = None,
) -> Tuple[Dict[str, UploadFile], Dict[str, Any]]:
    """
    Parse multipart form data separating images from configuration.

    Args:
        form_data: The multipart form data from FastAPI
        config_field: Name of the field containing JSON configuration (default: "config")
        image_fields: Optional list of expected image field names. If None, auto-detect.

    Returns:
        Tuple of (images_dict, config_dict):
        - images_dict: {field_name: UploadFile} for all image uploads
        - config_dict: Parsed configuration dictionary

    Example:
        images, config = await parse_multipart_with_images(await request.form())
        # images = {"image": <UploadFile>, "image2": <UploadFile>}
        # config = {"confidence": 0.5, "iou_threshold": 0.4}
    """
    images = {}
    config = {}

    # Identify image content types
    image_content_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
        "image/gif",
        "image/webp",
        "image/tiff",
        "application/octet-stream",  # Sometimes used for images
    }

    for field_name, field_value in form_data.items():
        if field_name == config_field:
            # Parse configuration JSON
            if isinstance(field_value, str):
                try:
                    config = json.loads(field_value)
                except json.JSONDecodeError as e:
                    raise InputImageLoadError(
                        message=f"Invalid JSON in '{config_field}' field: {e}",
                        public_message=f"Invalid JSON in configuration field",
                    )
            elif isinstance(field_value, UploadFile):
                # Config provided as file
                content = await field_value.read()
                try:
                    config = json.loads(content)
                except json.JSONDecodeError as e:
                    raise InputImageLoadError(
                        message=f"Invalid JSON in '{config_field}' file: {e}",
                        public_message=f"Invalid JSON in configuration file",
                    )
            else:
                logger.warning(f"Unexpected type for config field: {type(field_value)}")

        elif isinstance(field_value, UploadFile):
            # Check if it's an image by content type or field name
            content_type = field_value.content_type or ""
            is_image = (
                content_type.lower() in image_content_types
                or field_name.lower().startswith("image")
                or (image_fields and field_name in image_fields)
            )

            if is_image:
                images[field_name] = field_value
            else:
                logger.debug(f"Ignoring non-image upload field: {field_name}")

        else:
            # Regular form field - add to config if not config field
            if field_name != config_field:
                # Try to parse as JSON, otherwise use as string
                try:
                    config[field_name] = json.loads(field_value)
                except (json.JSONDecodeError, TypeError):
                    config[field_name] = field_value

    logger.debug(f"Parsed multipart: {len(images)} images, config keys: {list(config.keys())}")
    return images, config


async def upload_file_to_inference_request_image(
    upload_file: UploadFile,
    image_type: str = "numpy",
) -> InferenceRequestImage:
    """
    Convert an UploadFile to an InferenceRequestImage.

    Note: This keeps the image as raw bytes to avoid base64 encoding overhead.
    The image will be decoded directly to numpy array when needed.

    Args:
        upload_file: The uploaded file from multipart form
        image_type: Type hint for the image (default: "numpy")

    Returns:
        InferenceRequestImage with raw bytes
    """
    # Read the raw bytes
    image_bytes = await upload_file.read()

    if not image_bytes:
        raise InputImageLoadError(
            message=f"Empty image file: {upload_file.filename}",
            public_message="Uploaded image file is empty",
        )

    # Return as InferenceRequestImage with raw bytes
    # The image loading logic will handle decoding these bytes directly
    return InferenceRequestImage(
        type="numpy",  # Will be decoded to numpy array
        value=image_bytes,  # Raw bytes, not base64 encoded
    )


async def parse_workflow_multipart(
    form_data: FormData,
    inputs_field: str = "inputs",
    specification_field: str = "specification",
) -> Tuple[Dict[str, UploadFile], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Parse multipart form data for workflow requests.

    This handles the convention-based matching where multipart field names
    correspond to workflow input names.

    Args:
        form_data: The multipart form data from FastAPI
        inputs_field: Name of field containing workflow inputs JSON (default: "inputs")
        specification_field: Name of field containing workflow spec (default: "specification")

    Returns:
        Tuple of (images_dict, inputs_dict, specification_dict):
        - images_dict: {input_name: UploadFile} for image inputs
        - inputs_dict: Non-image workflow inputs
        - specification_dict: Workflow specification (if provided)

    Example:
        For a workflow with inputs ["image", "confidence", "prompt"]:

        Multipart parts:
        - "inputs": '{"confidence": 0.5, "prompt": "detect people"}'
        - "image": <binary image data>

        Returns:
        - images = {"image": <UploadFile>}
        - inputs = {"confidence": 0.5, "prompt": "detect people"}
        - specification = None
    """
    images = {}
    inputs = {}
    specification = None

    # Identify image content types
    image_content_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
        "image/gif",
        "image/webp",
        "image/tiff",
        "application/octet-stream",
    }

    for field_name, field_value in form_data.items():
        if field_name == inputs_field:
            # Parse inputs JSON
            if isinstance(field_value, str):
                try:
                    inputs = json.loads(field_value)
                except json.JSONDecodeError as e:
                    raise InputImageLoadError(
                        message=f"Invalid JSON in '{inputs_field}' field: {e}",
                        public_message="Invalid JSON in inputs field",
                    )
            elif isinstance(field_value, UploadFile):
                content = await field_value.read()
                try:
                    inputs = json.loads(content)
                except json.JSONDecodeError as e:
                    raise InputImageLoadError(
                        message=f"Invalid JSON in '{inputs_field}' file: {e}",
                        public_message="Invalid JSON in inputs file",
                    )

        elif field_name == specification_field:
            # Parse specification JSON
            if isinstance(field_value, str):
                try:
                    specification = json.loads(field_value)
                except json.JSONDecodeError as e:
                    raise InputImageLoadError(
                        message=f"Invalid JSON in '{specification_field}' field: {e}",
                        public_message="Invalid JSON in specification field",
                    )
            elif isinstance(field_value, UploadFile):
                content = await field_value.read()
                try:
                    specification = json.loads(content)
                except json.JSONDecodeError as e:
                    raise InputImageLoadError(
                        message=f"Invalid JSON in '{specification_field}' file: {e}",
                        public_message="Invalid JSON in specification file",
                    )

        elif isinstance(field_value, UploadFile):
            # Potential image field - use field name as input name
            content_type = field_value.content_type or ""
            is_image = (
                content_type.lower() in image_content_types
                or field_name.lower().startswith("image")
                or "image" in field_name.lower()
            )

            if is_image:
                images[field_name] = field_value
                logger.debug(f"Found image field: {field_name}")
            else:
                logger.debug(f"Ignoring non-image upload field: {field_name}")

        else:
            # Regular form field - could be a workflow input
            logger.debug(f"Found regular form field: {field_name} = {field_value}")

    logger.debug(
        f"Parsed workflow multipart: {len(images)} images, "
        f"{len(inputs)} inputs, "
        f"specification: {specification is not None}"
    )

    return images, inputs, specification


async def merge_images_into_inputs(
    images: Dict[str, UploadFile],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge image UploadFiles into workflow inputs dictionary.

    This implements the convention-based matching: if an input is not provided
    in the inputs dict, but a matching image field exists, use that image.

    Args:
        images: Dictionary of {field_name: UploadFile}
        inputs: Dictionary of workflow inputs

    Returns:
        Updated inputs dictionary with images merged in

    Example:
        images = {"image": <UploadFile>, "mask": <UploadFile>}
        inputs = {"confidence": 0.5}

        Result:
        {
            "confidence": 0.5,
            "image": InferenceRequestImage(...),
            "mask": InferenceRequestImage(...)
        }
    """
    merged_inputs = inputs.copy()

    for field_name, upload_file in images.items():
        # Convert UploadFile to InferenceRequestImage
        image_bytes = await upload_file.read()

        if not image_bytes:
            logger.warning(f"Empty image for field: {field_name}")
            continue

        # Check if this input already has a value
        if field_name in merged_inputs:
            existing_value = merged_inputs[field_name]

            # If existing value is already an image dict, skip
            if isinstance(existing_value, dict) and "type" in existing_value:
                logger.debug(f"Input '{field_name}' already has image data, skipping multipart image")
                continue

        # Add the image to inputs
        merged_inputs[field_name] = {
            "type": "numpy",
            "value": image_bytes,
        }
        logger.debug(f"Merged image '{field_name}' into inputs")

    return merged_inputs
