"""Description of the image embedding space CLIP models produce.

``ClipOnnx`` and ``ClipTorch`` share one preprocessing pipeline
(``create_clip_preprocessor``) and return the raw output of the visual tower,
so both describe their embeddings through the helpers below.
"""

from typing import Optional

from inference_models.errors import ModelInputError
from inference_models.models.clip.preprocessing import MEAN, STD

CLIP_IMAGE_FEATURE_DEFINITION = "clip-image-embeds@v1"
FEATURE_VECTOR_OUTPUT_TYPE = "feature_vector"


def describe_clip_image_embeddings(output_type: Optional[str] = None) -> dict:
    """Describe the representation CLIP's ``embed_images`` returns.

    Args:
        output_type: Requested representation. CLIP has no classification
            head, so only ``feature_vector`` (or None, meaning the same) exists.

    Returns:
        Feature definition, output type and normalization of the vectors.

    Raises:
        ModelInputError: If ``output_type`` is not ``feature_vector``.
    """
    if output_type not in (None, FEATURE_VECTOR_OUTPUT_TYPE):
        raise ModelInputError(
            message=f"CLIP produces only `{FEATURE_VECTOR_OUTPUT_TYPE}` image "
            f"embeddings, requested `{output_type}`.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )

    info = {
        "feature_definition": CLIP_IMAGE_FEATURE_DEFINITION,
        "output_type": FEATURE_VECTOR_OUTPUT_TYPE,
        "normalization": "none",
    }

    return info


def describe_clip_preprocessing(image_size: int) -> dict:
    """Describe the preprocessing CLIP's ``embed_images`` applies.

    The shorter image side is resized to ``image_size`` with bicubic
    interpolation, the image is center cropped to a square of that size,
    rescaled to [0, 1] and normalized with CLIP's mean and standard deviation.

    Args:
        image_size: Input size of the visual tower, in pixels.

    Returns:
        JSON-serializable preprocessing parameters.
    """
    preprocessing = {
        "image_size": image_size,
        "resize": "shortest_edge",
        "resample": "bicubic",
        "center_crop": True,
        "rescale_factor": 1 / 255,
        "image_mean": list(MEAN),
        "image_std": list(STD),
    }

    return preprocessing
