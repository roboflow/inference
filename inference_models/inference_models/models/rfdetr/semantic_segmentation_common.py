"""Loading and post-processing shared by the RF-DETR semantic segmentation backends."""

from typing import List, Optional, Tuple, Union

import torch

from inference_models.configuration import (
    INFERENCE_MODELS_RFDETR_SEMANTIC_SEGMENTATION_DEFAULT_CONFIDENCE,
)
from inference_models.entities import Confidence
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import (
    post_process_semantic_segmentation_logits,
)
from inference_models.weights_providers.entities import RecommendedParameters


def parse_semantic_segmentation_inference_config(
    config_path: str,
    *,
    backend_name: str,
    max_allowed_input_size: Optional[Union[int, Tuple[int, int]]],
) -> InferenceConfig:
    """Parse the inference config of an RF-DETR semantic segmentation package.

    Args:
        config_path: Path to ``inference_config.json``.
        backend_name: Backend named in the ``fit-longer-edge`` substitution warning.
        max_allowed_input_size: Largest input size the environment accepts.

    Returns:
        The parsed config, with ``fit-longer-edge`` replaced by ``stretch``.
    """
    inference_config = parse_inference_config(
        config_path=config_path,
        allowed_resize_modes={
            ResizeMode.STRETCH_TO,
            ResizeMode.LETTERBOX,
            ResizeMode.CENTER_CROP,
            ResizeMode.LETTERBOX_REFLECT_EDGES,
        },
        implicit_resize_mode_substitutions={
            ResizeMode.FIT_LONGER_EDGE: (
                ResizeMode.STRETCH_TO,
                None,
                f"RFDetr Semantic Segmentation model running with {backend_name} backend was trained "
                "with `fit-longer-edge` input resize mode. This transform cannot be applied properly "
                "for RFDetr models. To ensure interoperability, `stretch` resize mode will be used "
                "instead. If model was trained on Roboflow platform, we recommend using preprocessing "
                "method different that `fit-longer-edge`.",
            )
        },
        max_allowed_input_size=max_allowed_input_size,
    )
    return inference_config


def post_process_semantic_segmentation_results(
    model_results: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    *,
    class_names: List[str],
    background_class_id: int,
    device: torch.device,
    confidence: Confidence,
    recommended_parameters: Optional[RecommendedParameters],
) -> List[SemanticSegmentationResult]:
    """Turn RF-DETR semantic segmentation logits into class and confidence maps.

    Confidence is the top per-class sigmoid, the score the model was trained and
    evaluated with (per-class BCE), so recommended thresholds keep their meaning.

    Args:
        model_results: Class logits ``[B, K, H / 4, W / 4]``.
        pre_processing_meta: Metadata of each pre-processed image.
        class_names: Class names by class index, background included.
        background_class_id: Index of the background class.
        device: Device of the results.
        confidence: Confidence threshold, or ``"default"``.
        recommended_parameters: Thresholds recommended for the model.

    Returns:
        One result per image, at the original image size.
    """
    padding_frame_meta = [
        _with_padding_frame_as_inference_size(image_meta)
        for image_meta in pre_processing_meta
    ]
    results = post_process_semantic_segmentation_logits(
        model_results=model_results,
        pre_processing_meta=padding_frame_meta,
        class_names=class_names,
        background_class_id=background_class_id,
        device=device,
        confidence=confidence,
        recommended_parameters=recommended_parameters,
        default_confidence=INFERENCE_MODELS_RFDETR_SEMANTIC_SEGMENTATION_DEFAULT_CONFIDENCE,
        class_activation="sigmoid",
    )
    return results


def _with_padding_frame_as_inference_size(
    image_meta: PreProcessingMetadata,
) -> PreProcessingMetadata:
    # With a two-step resize, the letterbox pads are in the intermediate image,
    # which RF-DETR then stretches to the network input size.
    if image_meta.nonsquare_intermediate_size is None:
        return image_meta

    padding_frame_meta = image_meta._replace(
        inference_size=image_meta.nonsquare_intermediate_size,
        nonsquare_intermediate_size=None,
    )
    return padding_frame_meta
