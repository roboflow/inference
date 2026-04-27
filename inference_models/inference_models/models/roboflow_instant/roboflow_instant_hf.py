import os.path
from pickle import UnpicklingError
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_ROBOFLOW_INSTANT_MAX_DETECTIONS,
)
from inference_models.entities import Confidence, ImageDimensions
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.auto_loaders.entities import AnyModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.owlv2.entities import (
    ImageEmbeddings,
    ReferenceExamplesEmbeddings,
)
from inference_models.models.owlv2.owlv2_hf import OWLv2HF
from inference_models.weights_providers.entities import RecommendedParameters


class RoboflowInstantHF(ObjectDetectionModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        model_dependencies: Optional[Dict[str, AnyModel]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "ObjectDetectionModel":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["weights.pt"],
        )
        model_dependencies = model_dependencies or {}
        if "feature_extractor" in model_dependencies:
            feature_extractor: OWLv2HF = model_dependencies["feature_extractor"]
        else:
            feature_extractor = OWLv2HF.from_pretrained(
                os.path.join(
                    model_name_or_path, "model_dependencies", "feature_extractor"
                ),
                **kwargs,
            )
        try:
            weights_dict = torch.load(
                model_package_content["weights.pt"],
                map_location=device,
                weights_only=True,
            )
        except UnpicklingError as error:
            raise CorruptedModelPackageError(
                message="Could not deserialize RF Instant model weights. Contact Roboflow to get help.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        if "class_names" not in weights_dict or "train_data_dict" not in weights_dict:
            raise CorruptedModelPackageError(
                message="Corrupted weights of Roboflow Instant model detected. Contact Roboflow to get help.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        class_names = weights_dict["class_names"]
        train_data_dict = weights_dict["train_data_dict"]
        try:
            reference_examples_embeddings = (
                ReferenceExamplesEmbeddings.from_class_embeddings_dict(
                    class_embeddings=train_data_dict,
                    device=device,
                )
            )
        except Exception as error:
            raise CorruptedModelPackageError(
                message="Could not decode RF Instant model weights. Contact Roboflow to get help.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        return cls(
            feature_extractor=feature_extractor,
            class_names=class_names,
            reference_examples_embeddings=reference_examples_embeddings,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        feature_extractor: OWLv2HF,
        class_names: List[str],
        reference_examples_embeddings: ReferenceExamplesEmbeddings,
        recommended_parameters=None,
    ):
        self._feature_extractor = feature_extractor
        self._class_names = class_names
        self._reference_examples_embeddings = reference_examples_embeddings
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        max_detections: int = 300,
        **kwargs,
    ) -> Tuple[List[ImageEmbeddings], List[ImageDimensions]]:
        images_embeddings, images_dimensions = self._feature_extractor.embed_images(
            images=images,
            max_detections=max_detections,
        )
        return images_embeddings, images_dimensions

    def forward(
        self,
        pre_processed_images: List[ImageEmbeddings],
        confidence: float = INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_CONFIDENCE,
        iou_threshold: float = INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_IOU_THRESHOLD,
        **kwargs,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self._feature_extractor.forward_pass_with_precomputed_embeddings(
            images_embeddings=pre_processed_images,
            class_embeddings=self._reference_examples_embeddings.class_embeddings,
            confidence=confidence,
            iou_threshold=iou_threshold,
        )

    def post_process(
        self,
        model_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        pre_processing_meta: List[ImageDimensions],
        confidence: Confidence = "default",
        max_detections: int = INFERENCE_MODELS_ROBOFLOW_INSTANT_MAX_DETECTIONS,
        iou_threshold: float = INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_IOU_THRESHOLD,
        **kwargs,
    ) -> List[Detections]:
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_CONFIDENCE,
        )
        results = (
            self._feature_extractor.post_process_predictions_for_precomputed_embeddings(
                predictions=model_results,
                images_dimensions=pre_processing_meta,
                max_detections=max_detections,
                iou_threshold=iou_threshold,
            )
        )
        threshold_cpu = confidence_filter.get_threshold(self.class_names)
        refined = []
        for r in results:
            if isinstance(threshold_cpu, torch.Tensor):
                threshold = threshold_cpu.to(
                    dtype=r.confidence.dtype, device=r.confidence.device
                )
                keep = r.confidence >= threshold[r.class_id.long()]
            else:
                keep = r.confidence >= threshold_cpu
            refined.append(
                Detections(
                    xyxy=r.xyxy[keep],
                    class_id=r.class_id[keep],
                    confidence=r.confidence[keep],
                )
            )
        return refined
