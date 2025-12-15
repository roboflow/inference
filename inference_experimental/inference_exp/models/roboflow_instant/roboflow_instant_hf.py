import os.path
from pickle import UnpicklingError
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, ObjectDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ImageDimensions
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.auto_loaders.entities import AnyModel
from inference_exp.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.owlv2.entities import (
    ImageEmbeddings,
    ReferenceExamplesEmbeddings,
)
from inference_exp.models.owlv2.owlv2_hf import OWLv2HF


class RoboflowInstantHF(ObjectDetectionModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        model_dependencies: Optional[Dict[str, AnyModel]] = None,
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
                help_url="https://todo",
            ) from error
        if "class_names" not in weights_dict or "train_data_dict" not in weights_dict:
            raise CorruptedModelPackageError(
                message="Corrupted weights of Roboflow Instant model detected. Contact Roboflow to get help.",
                help_url="https://todo",
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
                help_url="https://todo",
            ) from error
        return cls(
            feature_extractor=feature_extractor,
            class_names=class_names,
            reference_examples_embeddings=reference_examples_embeddings,
        )

    def __init__(
        self,
        feature_extractor: OWLv2HF,
        class_names: List[str],
        reference_examples_embeddings: ReferenceExamplesEmbeddings,
    ):
        self._feature_extractor = feature_extractor
        self._class_names = class_names
        self._reference_examples_embeddings = reference_examples_embeddings

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[List[ImageEmbeddings], List[ImageDimensions]]:
        images_embeddings, images_dimensions = self._feature_extractor.embed_images(
            images=images
        )
        return images_embeddings, images_dimensions

    def forward(
        self,
        pre_processed_images: List[ImageEmbeddings],
        confidence_threshold: float = 0.99,
        iou_threshold: float = 0.3,
        **kwargs,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self._feature_extractor.forward_pass_with_precomputed_embeddings(
            images_embeddings=pre_processed_images,
            class_embeddings=self._reference_examples_embeddings.class_embeddings,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

    def post_process(
        self,
        model_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        pre_processing_meta: List[ImageDimensions],
        max_detections: int = 300,
        iou_threshold: float = 0.3,
        **kwargs,
    ) -> List[Detections]:
        return (
            self._feature_extractor.post_process_predictions_for_precomputed_embeddings(
                predictions=model_results,
                images_dimensions=pre_processing_meta,
                max_detections=max_detections,
                iou_threshold=iou_threshold,
            )
        )
