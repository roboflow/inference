from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from inference_exp import Detections, OpenVocabularyObjectDetectionModel
from inference_exp.configuration import (
    ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA,
    ALLOW_NON_HTTPS_URL_INPUT,
    ALLOW_URL_INPUT,
    ALLOW_URL_INPUT_WITHOUT_FQDN,
    BLACKLISTED_DESTINATIONS_FOR_URL_INPUT,
    DEFAULT_DEVICE,
    WHITELISTED_DESTINATIONS_FOR_URL_INPUT,
)
from inference_exp.entities import ImageDimensions
from inference_exp.errors import ModelInputError
from inference_exp.models.base.types import PreprocessedInputs, PreprocessingMetadata
from inference_exp.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
)
from inference_exp.models.owlv2.cache import (
    OwlV2ClassEmbeddingsCache,
    OwlV2ClassEmbeddingsCacheNullObject,
    OwlV2ImageEmbeddingsCache,
    OwlV2ImageEmbeddingsCacheNullObject,
    hash_reference_examples,
)
from inference_exp.models.owlv2.entities import (
    NEGATIVE_EXAMPLE,
    POSITIVE_EXAMPLE,
    ImageEmbeddings,
    LazyReferenceExample,
    ReferenceExample,
    ReferenceExamplesClassEmbeddings,
    ReferenceExamplesEmbeddings,
)
from inference_exp.models.owlv2.reference_dataset import (
    LazyImageWrapper,
    compute_image_hash,
)
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.models.owlv2.modeling_owlv2 import Owlv2ObjectDetectionOutput, box_iou

Query = Dict[
    str,
    Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]],
]


class OWLv2HF(
    OpenVocabularyObjectDetectionModel[
        torch.Tensor, List[ImageDimensions], Owlv2ObjectDetectionOutput
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        owlv2_class_embeddings_cache: Optional[OwlV2ClassEmbeddingsCache] = None,
        owlv2_images_embeddings_cache: Optional[OwlV2ImageEmbeddingsCache] = None,
        allow_url_input: bool = ALLOW_URL_INPUT,
        allow_non_https_url: bool = ALLOW_NON_HTTPS_URL_INPUT,
        allow_url_without_fqdn: bool = ALLOW_URL_INPUT_WITHOUT_FQDN,
        whitelisted_domains: Optional[List[str]] = None,
        blacklisted_domains: Optional[List[str]] = None,
        allow_local_storage_access_for_reference_images: bool = ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA,
        owlv2_enforce_model_compilation: bool = False,
        **kwargs,
    ) -> "OpenVocabularyObjectDetectionModel":
        if owlv2_class_embeddings_cache is None:
            owlv2_class_embeddings_cache = OwlV2ClassEmbeddingsCacheNullObject()
        if owlv2_images_embeddings_cache is None:
            owlv2_images_embeddings_cache = OwlV2ImageEmbeddingsCacheNullObject()
        if whitelisted_domains is None:
            whitelisted_domains = WHITELISTED_DESTINATIONS_FOR_URL_INPUT
        if blacklisted_domains is None:
            blacklisted_domains = BLACKLISTED_DESTINATIONS_FOR_URL_INPUT
        processor = Owlv2Processor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            use_fast=True,
        )
        model = Owlv2ForObjectDetection.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).to(device)
        instance = cls(
            model=model,
            processor=processor,
            device=device,
            owlv2_class_embeddings_cache=owlv2_class_embeddings_cache,
            owlv2_images_embeddings_cache=owlv2_images_embeddings_cache,
            allow_url_input=allow_url_input,
            allow_non_https_url=allow_non_https_url,
            allow_url_without_fqdn=allow_url_without_fqdn,
            whitelisted_domains=whitelisted_domains,
            blacklisted_domains=blacklisted_domains,
            allow_local_storage_access_for_reference_images=allow_local_storage_access_for_reference_images,
        )
        if owlv2_enforce_model_compilation:
            instance.optimize_for_inference()
        return instance

    def __init__(
        self,
        model: Owlv2ForObjectDetection,
        processor: Owlv2Processor,
        device: torch.device,
        owlv2_class_embeddings_cache: OwlV2ClassEmbeddingsCache,
        owlv2_images_embeddings_cache: OwlV2ImageEmbeddingsCache,
        allow_url_input: bool,
        allow_non_https_url: bool,
        allow_url_without_fqdn: bool,
        whitelisted_domains: Optional[List[str]],
        blacklisted_domains: Optional[List[str]],
        allow_local_storage_access_for_reference_images: bool,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._owlv2_class_embeddings_cache = owlv2_class_embeddings_cache
        self._owlv2_images_embeddings_cache = owlv2_images_embeddings_cache
        self._allow_url_input = allow_url_input
        self._allow_non_https_url = allow_non_https_url
        self._allow_url_without_fqdn = allow_url_without_fqdn
        self._whitelisted_domains = whitelisted_domains
        self._blacklisted_domains = blacklisted_domains
        self._allow_local_storage_access_for_reference_images = (
            allow_local_storage_access_for_reference_images
        )
        self._compiled = False

    def optimize_for_inference(self) -> None:
        if self._compiled:
            return None
        self._model.owlv2.vision_model = torch.compile(self._model.owlv2.vision_model)
        example_image = torch.randint(
            low=0, high=255, size=(3, 128, 128), dtype=torch.uint8
        ).to(self._device)
        _ = self.infer(example_image, ["some", "other"])
        self._compiled = True

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        image_dimensions = extract_input_images_dimensions(images=images)
        inputs = self._processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self._device), image_dimensions

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        classes: List[str],
        **kwargs,
    ) -> Owlv2ObjectDetectionOutput:
        input_ids = self._processor(text=[classes], return_tensors="pt")[
            "input_ids"
        ].to(self._device)
        with torch.inference_mode():
            return self._model(input_ids=input_ids, pixel_values=pre_processed_images)

    def post_process(
        self,
        model_results: Owlv2ObjectDetectionOutput,
        pre_processing_meta: List[ImageDimensions],
        conf_thresh: float = 0.1,
        iou_thresh: float = 0.45,
        class_agnostic: bool = False,
        max_detections: int = 100,
        **kwargs,
    ) -> List[Detections]:
        target_sizes = [(dim.height, dim.width) for dim in pre_processing_meta]
        post_processed_outputs = self._processor.post_process_grounded_object_detection(
            outputs=model_results,
            target_sizes=target_sizes,
            threshold=conf_thresh,
        )
        results = []
        for i in range(len(post_processed_outputs)):
            boxes, scores, labels = (
                post_processed_outputs[i]["boxes"],
                post_processed_outputs[i]["scores"],
                post_processed_outputs[i]["labels"],
            )
            nms_class_ids = torch.zeros_like(labels) if class_agnostic else labels
            keep = torchvision.ops.batched_nms(boxes, scores, nms_class_ids, iou_thresh)
            keep = keep[:max_detections]
            results.append(
                Detections(
                    xyxy=boxes[keep].contiguous().int(),
                    class_id=labels[keep].contiguous().int(),
                    confidence=scores[keep].contiguous(),
                )
            )
        return results

    def infer_with_reference_examples(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        reference_examples: List[ReferenceExample],
        confidence_threshold: float = 0.99,
        iou_threshold: float = 0.3,
        max_detections: int = 300,
    ) -> List[Detections]:
        reference_embeddings = self.prepare_reference_examples_embeddings(
            reference_examples=reference_examples,
            iou_threshold=iou_threshold,
        )
        return self.infer_with_reference_examples_embeddings(
            images=images,
            class_embeddings=reference_embeddings.class_embeddings,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    def infer_with_reference_examples_embeddings(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        class_embeddings: Dict[str, ReferenceExamplesClassEmbeddings],
        confidence_threshold: float = 0.99,
        iou_threshold: float = 0.3,
        max_detections: int = 300,
    ) -> List[Detections]:
        images_embeddings, images_dimensions = self.embed_images(
            images=images, max_detections=max_detections
        )
        images_predictions = self.forward_pass_with_precomputed_embeddings(
            images_embeddings=images_embeddings,
            class_embeddings=class_embeddings,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        return self.post_process_predictions_for_precomputed_embeddings(
            predictions=images_predictions,
            images_dimensions=images_dimensions,
            max_detections=max_detections,
            iou_threshold=iou_threshold,
        )

    def forward_pass_with_precomputed_embeddings(
        self,
        images_embeddings: List[ImageEmbeddings],
        class_embeddings: Dict[str, ReferenceExamplesClassEmbeddings],
        confidence_threshold: float = 0.99,
        iou_threshold: float = 0.3,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        results = []
        for image_embedding in images_embeddings:
            image_embedding = image_embedding.to(self._device)
            class_mapping, class_names = make_class_mapping(
                class_names=class_embeddings.keys()
            )
            all_predicted_boxes, all_predicted_classes, all_predicted_scores = (
                [],
                [],
                [],
            )
            for (
                class_name,
                reference_examples_class_embeddings,
            ) in class_embeddings.items():
                boxes, classes, scores = get_class_predictions_from_embedings(
                    reference_examples_class_embeddings=reference_examples_class_embeddings,
                    image_class_embeddings=image_embedding.image_class_embeddings,
                    image_boxes=image_embedding.boxes,
                    confidence_threshold=confidence_threshold,
                    class_mapping=class_mapping,
                    class_name=class_name,
                    iou_threshold=iou_threshold,
                )
                all_predicted_boxes.append(boxes)
                all_predicted_classes.append(classes)
                all_predicted_scores.append(scores)
            if not all_predicted_boxes:
                results.append(
                    (torch.empty((0,)), torch.empty((0,)), torch.empty((0,)))
                )
                continue
            all_predicted_boxes = torch.cat(all_predicted_boxes, dim=0)
            all_predicted_classes = torch.cat(all_predicted_classes, dim=0)
            all_predicted_scores = torch.cat(all_predicted_scores, dim=0)
            results.append(
                (all_predicted_boxes, all_predicted_classes, all_predicted_scores)
            )
        return results

    def post_process_predictions_for_precomputed_embeddings(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        images_dimensions: List[ImageDimensions],
        max_detections: int = 300,
        iou_threshold: float = 0.3,
    ) -> List[Detections]:
        results = []
        for image_predictions, image_dimensions in zip(predictions, images_dimensions):
            all_predicted_boxes, all_predicted_classes, all_predicted_scores = (
                image_predictions
            )
            if all_predicted_boxes.numel() == 0:
                results.append(
                    Detections(
                        xyxy=torch.empty(
                            (0, 4), dtype=torch.int32, device=self._device
                        ),
                        confidence=torch.empty(
                            (0,), dtype=torch.float32, device=self._device
                        ),
                        class_id=torch.empty(
                            (0,), dtype=torch.int32, device=self._device
                        ),
                    )
                )
                continue
            survival_indices = torchvision.ops.nms(
                to_corners(all_predicted_boxes), all_predicted_scores, iou_threshold
            )
            all_predicted_boxes = all_predicted_boxes[survival_indices]
            all_predicted_classes = all_predicted_classes[survival_indices]
            all_predicted_scores = all_predicted_scores[survival_indices]
            if len(all_predicted_boxes) > max_detections:
                all_predicted_boxes = all_predicted_boxes[:max_detections]
                all_predicted_classes = all_predicted_classes[:max_detections]
                all_predicted_scores = all_predicted_scores[:max_detections]
            xyxy = xywh_normalized_to_xyxy(
                boxes_xywh=all_predicted_boxes,
                image_size_wh=(image_dimensions.width, image_dimensions.height),
            )
            results.append(
                Detections(
                    xyxy=xyxy.int(),
                    confidence=all_predicted_scores,
                    class_id=all_predicted_classes.int(),
                )
            )
        return results

    def prepare_reference_examples_embeddings(
        self,
        reference_examples: List[ReferenceExample],
        iou_threshold: float,
        return_image_embeddings: bool = False,
    ) -> ReferenceExamplesEmbeddings:
        lazy_reference_examples = [
            LazyReferenceExample(
                image=LazyImageWrapper.init(
                    image=example.image,
                    allow_url_input=self._allow_url_input,
                    allow_non_https_url=self._allow_non_https_url,
                    allow_url_without_fqdn=self._allow_url_without_fqdn,
                    whitelisted_domains=self._whitelisted_domains,
                    blacklisted_domains=self._blacklisted_domains,
                    allow_local_storage_access=self._allow_local_storage_access_for_reference_images,
                ),
                boxes=example.boxes,
            )
            for example in reference_examples
        ]
        examples_hash_key = hash_reference_examples(
            reference_examples=lazy_reference_examples
        )
        cached_embeddings = self._owlv2_class_embeddings_cache.retrieve_embeddings(
            key=examples_hash_key
        )
        if cached_embeddings is not None and not return_image_embeddings:
            cached_embeddings = {
                k: v.to(self._device) for k, v in cached_embeddings.items()
            }
            return ReferenceExamplesEmbeddings(
                class_embeddings=cached_embeddings,
                image_embeddings=None,
            )
        class_embeddings_dict = defaultdict(
            lambda: {POSITIVE_EXAMPLE: [], NEGATIVE_EXAMPLE: []}
        )
        bool_to_literal = {True: POSITIVE_EXAMPLE, False: NEGATIVE_EXAMPLE}
        image_embeddings_to_be_returned = {}
        for reference_example in lazy_reference_examples:
            image_embeddings = self.embed_image(image=reference_example.image)
            if return_image_embeddings:
                image_embeddings_to_be_returned[image_embeddings.image_hash] = (
                    image_embeddings
                )
            coordinates = [
                bbox.to_tuple(image_wh=image_embeddings.image_size_wh)
                for bbox in reference_example.boxes
            ]
            classes = [box.cls for box in reference_example.boxes]
            is_positive = [not box.negative for box in reference_example.boxes]
            query = {image_embeddings.image_hash: coordinates}
            image_class_embeddings_matching_query = self.query_images_for_bboxes(
                query=query,
                images_embeddings={image_embeddings.image_hash: image_embeddings},
                iou_threshold=iou_threshold,
            )
            if image_class_embeddings_matching_query is None:
                continue
            for embedding, class_name, is_pos in zip(
                image_class_embeddings_matching_query, classes, is_positive
            ):
                class_embeddings_dict[class_name][bool_to_literal[is_pos]].append(
                    embedding
                )
        class_embeddings = {
            class_name: ReferenceExamplesClassEmbeddings(
                positive=(
                    torch.stack(embeddings[POSITIVE_EXAMPLE])
                    if embeddings[POSITIVE_EXAMPLE]
                    else None
                ),
                negative=(
                    torch.stack(embeddings[NEGATIVE_EXAMPLE])
                    if embeddings[NEGATIVE_EXAMPLE]
                    else None
                ),
            )
            for class_name, embeddings in class_embeddings_dict.items()
        }
        self._owlv2_class_embeddings_cache.save_embeddings(
            key=examples_hash_key, embeddings=class_embeddings
        )
        return ReferenceExamplesEmbeddings(
            class_embeddings=class_embeddings,
            image_embeddings=(
                image_embeddings_to_be_returned if return_image_embeddings else None
            ),
        )

    @torch.inference_mode()
    def query_images_for_bboxes(
        self,
        query: Query,
        images_embeddings: Dict[str, ImageEmbeddings],
        iou_threshold: float,
    ) -> Optional[torch.Tensor]:
        query_embeddings = []
        for image_hash, query_boxes in query.items():
            image_embeddings = images_embeddings.get(image_hash)
            if image_embeddings is None:
                raise ModelInputError(
                    message="Could not find image embeddings matching bounding boxes query for OWLv2 model. This "
                    "means that most likely, model API was used incorrectly.",
                    help_url="https://todo",
                )
            image_embeddings = image_embeddings.to(self._device)
            query_boxes_tensor = torch.tensor(
                query_boxes,
                dtype=image_embeddings.boxes.dtype,
                device=self._device,
            )
            if image_embeddings.boxes.numel() == 0 or query_boxes_tensor.numel() == 0:
                continue
            iou, _ = box_iou(
                boxes1=to_corners(image_embeddings.boxes),
                boxes2=to_corners(query_boxes_tensor),
            )  # 3000, k
            ious, indices = torch.max(iou, dim=0)
            # filter for only iou > 0.4
            iou_mask = ious > iou_threshold
            indices = indices[iou_mask]
            if not indices.numel() > 0:
                continue
            matching_image_embeddings = image_embeddings.image_class_embeddings[indices]
            query_embeddings.append(matching_image_embeddings)
        if not query_embeddings:
            return None
        return torch.cat(query_embeddings, dim=0)

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        max_detections: int = 300,
    ) -> Tuple[List[ImageEmbeddings], List[ImageDimensions]]:
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 3:
                images = [images]
            else:
                images = torch.unbind(images, dim=0)
        elif not isinstance(images, list):
            images = [images]
        results = []
        image_dimensions = []
        for image in images:
            image_embedding = self.embed_image(
                image=image, max_detections=max_detections
            )
            results.append(image_embedding)
            image_dimensions.append(
                ImageDimensions(
                    height=image_embedding.image_size_wh[1],
                    width=image_embedding.image_size_wh[0],
                )
            )
        return results, image_dimensions

    @torch.inference_mode()
    def embed_image(
        self,
        image: Union[torch.Tensor, np.ndarray, LazyImageWrapper],
        max_detections: int = 300,
        unload_after_use: bool = True,
    ) -> ImageEmbeddings:
        if isinstance(image, LazyImageWrapper):
            image_hash = image.get_hash()
            image_instance = image.as_numpy()
            if unload_after_use:
                image.unload_image()
        else:
            image_hash = compute_image_hash(image=image)
            image_instance = image
        cached_embeddings = self._owlv2_images_embeddings_cache.retrieve_embeddings(
            key=image_hash
        )
        if cached_embeddings:
            return cached_embeddings
        pixel_values, image_dimensions = self.pre_process(image_instance)
        device_type = self._device.type
        with torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"
        ):
            image_embeds, *_ = self._model.image_embedder(pixel_values=pixel_values)
            batch_size, h, w, dim = image_embeds.shape
            image_features = image_embeds.reshape(batch_size, h * w, dim)
            objectness = self._model.objectness_predictor(image_features)
            boxes = self._model.box_predictor(image_features, feature_map=image_embeds)
        image_class_embeddings = self._model.class_head.dense0(image_features)
        image_class_embeddings /= (
            torch.linalg.norm(image_class_embeddings, ord=2, dim=-1, keepdim=True)
            + 1e-6
        )
        logit_shift = self._model.class_head.logit_shift(image_features)
        logit_scale = (
            self._model.class_head.elu(
                self._model.class_head.logit_scale(image_features)
            )
            + 1
        )
        objectness = objectness.sigmoid()
        objectness, boxes, image_class_embeddings, logit_shift, logit_scale = (
            filter_tensors_by_objectness(
                objectness,
                boxes,
                image_class_embeddings,
                logit_shift,
                logit_scale,
                max_detections,
            )
        )
        embeddings = ImageEmbeddings(
            image_hash=image_hash,
            objectness=objectness,
            boxes=boxes,
            image_class_embeddings=image_class_embeddings,
            logit_shift=logit_shift,
            logit_scale=logit_scale,
            image_size_wh=(image_dimensions[0].width, image_dimensions[0].height),
        )
        self._owlv2_images_embeddings_cache.save_embeddings(embeddings=embeddings)
        return embeddings


def to_corners(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def make_class_mapping(
    class_names: Iterable[str],
) -> Tuple[Dict[Tuple[str, str], int], List[str]]:
    class_names = sorted(class_names)
    class_map_positive = {
        (class_name, POSITIVE_EXAMPLE): i for i, class_name in enumerate(class_names)
    }
    class_map_negative = {
        (class_name, NEGATIVE_EXAMPLE): i + len(class_names)
        for i, class_name in enumerate(class_names)
    }
    class_map = {**class_map_positive, **class_map_negative}
    return class_map, class_names


def filter_tensors_by_objectness(
    objectness: torch.Tensor,
    boxes: torch.Tensor,
    image_class_embeds: torch.Tensor,
    logit_shift: torch.Tensor,
    logit_scale: torch.Tensor,
    max_detections: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    objectness = objectness.squeeze(0)
    objectness, objectness_indices = torch.topk(objectness, max_detections, dim=0)
    boxes = boxes.squeeze(0)
    image_class_embeds = image_class_embeds.squeeze(0)
    logit_shift = logit_shift.squeeze(0).squeeze(1)
    logit_scale = logit_scale.squeeze(0).squeeze(1)
    boxes = boxes[objectness_indices]
    image_class_embeds = image_class_embeds[objectness_indices]
    logit_shift = logit_shift[objectness_indices]
    logit_scale = logit_scale[objectness_indices]
    return objectness, boxes, image_class_embeds, logit_shift, logit_scale


def get_class_predictions_from_embedings(
    reference_examples_class_embeddings: ReferenceExamplesClassEmbeddings,
    image_class_embeddings: torch.Tensor,
    image_boxes: torch.Tensor,
    confidence_threshold: float,
    class_mapping: Dict[Tuple[str, str], int],
    class_name: str,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_boxes_per_class = []
    predicted_class_indices_per_class = []
    predicted_scores_per_class = []
    positive_arr_per_class = []
    if reference_examples_class_embeddings.positive is not None:
        pred_logits = torch.einsum(
            "sd,nd->ns",
            image_class_embeddings,
            reference_examples_class_embeddings.positive,
        )
        prediction_scores = pred_logits.max(dim=0)[0]
        prediction_scores = (prediction_scores + 1) / 2
        score_mask = prediction_scores > confidence_threshold
        predicted_boxes_per_class.append(image_boxes[score_mask])
        scores = prediction_scores[score_mask]
        predicted_scores_per_class.append(scores)
        class_ind = class_mapping[(class_name, POSITIVE_EXAMPLE)]
        predicted_class_indices_per_class.append(class_ind * torch.ones_like(scores))
        positive_arr_per_class.append(torch.ones_like(scores))
    if reference_examples_class_embeddings.negative is not None:
        pred_logits = torch.einsum(
            "sd,nd->ns",
            image_class_embeddings,
            reference_examples_class_embeddings.positive,
        )
        prediction_scores = pred_logits.max(dim=0)[0]
        prediction_scores = (prediction_scores + 1) / 2
        score_mask = prediction_scores > confidence_threshold
        predicted_boxes_per_class.append(image_boxes[score_mask])
        scores = prediction_scores[score_mask]
        predicted_scores_per_class.append(scores)
        class_ind = class_mapping[(class_name, NEGATIVE_EXAMPLE)]
        predicted_class_indices_per_class.append(class_ind * torch.ones_like(scores))
        positive_arr_per_class.append(torch.zeros_like(scores))
    if not predicted_boxes_per_class:
        return (
            torch.empty((0, 4)),
            torch.empty((0,)),
            torch.empty((0,)),
        )
    # concat tensors
    pred_boxes = torch.cat(predicted_boxes_per_class, dim=0).float()
    pred_classes = torch.cat(predicted_class_indices_per_class, dim=0).float()
    pred_scores = torch.cat(predicted_scores_per_class, dim=0).float()
    positive = torch.cat(positive_arr_per_class, dim=0).float()
    # nms
    survival_indices = torchvision.ops.nms(
        to_corners(pred_boxes), pred_scores, iou_threshold
    )
    # filter to post-nms
    pred_boxes = pred_boxes[survival_indices, :]
    pred_classes = pred_classes[survival_indices]
    pred_scores = pred_scores[survival_indices]
    positive = positive[survival_indices]
    is_positive = positive == 1
    # return only positive elements of tensor
    return pred_boxes[is_positive], pred_classes[is_positive], pred_scores[is_positive]


def xywh_normalized_to_xyxy(
    boxes_xywh: torch.Tensor, image_size_wh: Tuple[int, int]
) -> torch.Tensor:
    max_dim = max(image_size_wh)
    x_center = boxes_xywh[..., 0] * max_dim
    y_center = boxes_xywh[..., 1] * max_dim
    box_width = boxes_xywh[..., 2] * max_dim
    box_height = boxes_xywh[..., 3] * max_dim
    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2
    return torch.stack([x1, y1, x2, y2], dim=-1).to(device=boxes_xywh.device)
