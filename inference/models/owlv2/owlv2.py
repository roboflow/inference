import gc
import hashlib
import os
import pickle
import weakref
from collections import defaultdict
from typing import Any, Dict, List, Literal, NewType, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.models.owlv2.modeling_owlv2 import box_iou

from inference.core import logger
from inference.core.cache.model_artifacts import save_bytes_in_cache
from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import (
    DEVICE,
    MAX_DETECTIONS,
    MODEL_CACHE_DIR,
    OWLV2_COMPILE_MODEL,
    OWLV2_CPU_IMAGE_CACHE_SIZE,
    OWLV2_IMAGE_CACHE_SIZE,
    OWLV2_MODEL_CACHE_SIZE,
    OWLV2_VERSION_ID,
)
from inference.core.exceptions import InvalidModelIDError, ModelArtefactError
from inference.core.models.roboflow import (
    DEFAULT_COLOR_PALETTE,
    RoboflowInferenceModel,
    draw_detection_predictions,
)
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_from_url,
    get_roboflow_instant_model_data,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import (
    ImageType,
    extract_image_payload_and_type,
    load_image_rgb,
)

CPU_IMAGE_EMBED_CACHE_SIZE = OWLV2_CPU_IMAGE_CACHE_SIZE

# TYPES
Hash = NewType("Hash", str)
PosNegKey = Literal["positive", "negative"]
PosNegDictType = Dict[PosNegKey, torch.Tensor]
QuerySpecType = Dict[Hash, List[List[int]]]
if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def to_corners(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


from collections import OrderedDict


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class Owlv2Singleton:
    _instances = weakref.WeakValueDictionary()

    def __new__(cls, huggingface_id: str):
        if huggingface_id not in cls._instances:
            logger.info(f"Creating new OWLv2 instance for {huggingface_id}")
            instance = super().__new__(cls)
            instance.huggingface_id = huggingface_id
            # Load model directly in the instance
            logger.info(f"Loading OWLv2 model from {huggingface_id}")
            model = (
                Owlv2ForObjectDetection.from_pretrained(huggingface_id)
                .eval()
                .to(DEVICE)
            )

            if OWLV2_COMPILE_MODEL:
                torch._dynamo.config.suppress_errors = True
                model.owlv2.vision_model = torch.compile(model.owlv2.vision_model)
            instance.model = model
            cls._instances[huggingface_id] = instance
        return cls._instances[huggingface_id]


def preprocess_image(
    np_image: np.ndarray,
    image_size: Tuple[int, int],
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
) -> torch.Tensor:
    """Preprocess an image for OWLv2 by resizing, normalizing, and padding it.
    This is much faster than using the Owlv2Processor directly, as we ensure we use GPU if available.

    Args:
        np_image (np.ndarray): The image to preprocess, with shape (H, W, 3)
        image_size (tuple[int, int]): The target size of the image
        image_mean (torch.Tensor): The mean of the image, on DEVICE, with shape (1, 3, 1, 1)
        image_std (torch.Tensor): The standard deviation of the image, on DEVICE, with shape (1, 3, 1, 1)

    Returns:
        torch.Tensor: The preprocessed image, on DEVICE, with shape (1, 3, H, W)
    """
    current_size = np_image.shape[:2]

    r = min(image_size[0] / current_size[0], image_size[1] / current_size[1])
    target_size = (int(r * current_size[0]), int(r * current_size[1]))

    torch_image = (
        torch.tensor(np_image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
        .to(dtype=torch.float32)
        / 255.0
    )
    torch_image = F.interpolate(
        torch_image, size=target_size, mode="bilinear", align_corners=False
    )

    padded_image_tensor = torch.ones((1, 3, *image_size), device=DEVICE) * 0.5
    padded_image_tensor[:, :, : torch_image.shape[2], : torch_image.shape[3]] = (
        torch_image
    )

    padded_image_tensor = (padded_image_tensor - image_mean) / image_std

    return padded_image_tensor


def filter_tensors_by_objectness(
    objectness: torch.Tensor,
    boxes: torch.Tensor,
    image_class_embeds: torch.Tensor,
    logit_shift: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    objectness = objectness.squeeze(0)
    objectness, objectness_indices = torch.topk(objectness, MAX_DETECTIONS, dim=0)
    boxes = boxes.squeeze(0)
    image_class_embeds = image_class_embeds.squeeze(0)
    logit_shift = logit_shift.squeeze(0).squeeze(1)
    logit_scale = logit_scale.squeeze(0).squeeze(1)
    boxes = boxes[objectness_indices]
    image_class_embeds = image_class_embeds[objectness_indices]
    logit_shift = logit_shift[objectness_indices]
    logit_scale = logit_scale[objectness_indices]
    return objectness, boxes, image_class_embeds, logit_shift, logit_scale


def get_class_preds_from_embeds(
    pos_neg_embedding_dict: PosNegDictType,
    image_class_embeds: torch.Tensor,
    confidence: float,
    image_boxes: torch.Tensor,
    class_map: Dict[Tuple[str, str], int],
    class_name: str,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_boxes_per_class = []
    predicted_class_indices_per_class = []
    predicted_scores_per_class = []
    positive_arr_per_class = []
    for positive, embedding in pos_neg_embedding_dict.items():
        if embedding is None:
            continue
        pred_logits = torch.einsum("sd,nd->ns", image_class_embeds, embedding)
        prediction_scores = pred_logits.max(dim=0)[0]
        prediction_scores = (prediction_scores + 1) / 2
        score_mask = prediction_scores > confidence
        predicted_boxes_per_class.append(image_boxes[score_mask])
        scores = prediction_scores[score_mask]
        predicted_scores_per_class.append(scores)
        class_ind = class_map[(class_name, positive)]
        predicted_class_indices_per_class.append(class_ind * torch.ones_like(scores))
        positive_arr_per_class.append(
            int(positive == "positive") * torch.ones_like(scores)
        )

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


def make_class_map(
    query_embeddings: Dict[str, PosNegDictType],
) -> Tuple[Dict[Tuple[str, str], int], List[str]]:
    class_names = sorted(list(query_embeddings.keys()))
    class_map_positive = {
        (class_name, "positive"): i for i, class_name in enumerate(class_names)
    }
    class_map_negative = {
        (class_name, "negative"): i + len(class_names)
        for i, class_name in enumerate(class_names)
    }
    class_map = {**class_map_positive, **class_map_negative}
    return class_map, class_names


def hash_function(value: Any) -> Hash:
    # wrapper so we can change the hashing function in the future
    return hashlib.sha1(value).hexdigest()


class LazyImageRetrievalWrapper:
    def __init__(self, image: Any):
        self.image = (
            image  # Prefer passing a file path or URL to avoid large in‑memory arrays.
        )
        self._image_as_numpy = None
        self._image_hash = None

    def unload_numpy_image(self):
        self._image_as_numpy = None
        if isinstance(self.image, np.ndarray):
            self.image = None  # Clear large array reference.

    @property
    def image_as_numpy(self) -> np.ndarray:
        if self._image_as_numpy is None:
            self._image_as_numpy = load_image_rgb(self.image)
        return self._image_as_numpy

    @property
    def image_hash(self) -> Hash:
        if self._image_hash is None:
            image_payload, image_type = extract_image_payload_and_type(self.image)
            if image_type in (ImageType.URL, ImageType.FILE):
                self._image_hash = image_payload
            elif image_type is ImageType.BASE64:
                if isinstance(image_payload, str):
                    image_payload = image_payload.encode("utf-8")
                self._image_hash = hash_function(image_payload)
            else:
                self._image_hash = hash_function(self.image_as_numpy.tobytes())
        return self._image_hash


def hash_wrapped_training_data(wrapped_training_data: List[Dict[str, Any]]) -> Hash:
    just_hash_relevant_data = [
        [
            d["image"].image_hash,
            d["boxes"],
        ]
        for d in wrapped_training_data
    ]
    # we dump to pickle to serialize the data as a single object
    return hash_function(pickle.dumps(just_hash_relevant_data))


class OwlV2(RoboflowInferenceModel):
    task_type = "object-detection"
    box_format = "xywh"

    def __init__(self, model_id=f"owlv2/{OWLV2_VERSION_ID}", *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)
        # TODO: owlv2 makes use of version_id - version_id is being dropped so this class needs to be refactored
        if self.version_id is None:
            owlv2_model_id_chunks = model_id.split("/")
            if len(owlv2_model_id_chunks) != 2:
                raise InvalidModelIDError(f"Model ID: `{model_id}` is invalid.")
            self.dataset_id = owlv2_model_id_chunks[0]
            self.version_id = owlv2_model_id_chunks[1]
        hf_id = os.path.join("google", self.version_id)
        processor = Owlv2Processor.from_pretrained(hf_id)
        self.image_size = tuple(processor.image_processor.size.values())
        self.image_mean = torch.tensor(
            processor.image_processor.image_mean, device=DEVICE
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            processor.image_processor.image_std, device=DEVICE
        ).view(1, 3, 1, 1)
        self.model = Owlv2Singleton(hf_id).model
        self.reset_cache()

    def reset_cache(self):
        # each entry should be on the order of 300*4KB, so 1000 is 400MB of CUDA memory
        self.image_embed_cache = LimitedSizeDict(size_limit=OWLV2_IMAGE_CACHE_SIZE)
        # no need for limit here, as we're only storing on CPU
        self.cpu_image_embed_cache = LimitedSizeDict(
            size_limit=CPU_IMAGE_EMBED_CACHE_SIZE
        )
        # each entry should be on the order of 10 bytes, so 1000 is 10KB
        self.image_size_cache = LimitedSizeDict(size_limit=OWLV2_IMAGE_CACHE_SIZE)
        # entry size will vary depending on the number of samples, but 10 should be safe
        self.class_embeddings_cache = LimitedSizeDict(size_limit=OWLV2_MODEL_CACHE_SIZE)

    def draw_predictions(
        self,
        inference_request,
        inference_response,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        all_class_names = [x.class_name for x in inference_response.predictions]
        all_class_names = sorted(list(set(all_class_names)))

        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors={
                class_name: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
                for (i, class_name) in enumerate(all_class_names)
            },
        )

    def download_weights(self) -> None:
        # Download from huggingface
        pass

    def get_image_embeds(self, image_hash: Hash) -> Optional[torch.Tensor]:
        if image_hash in self.image_embed_cache:
            return self.image_embed_cache[image_hash]
        elif image_hash in self.cpu_image_embed_cache:
            tensors = self.cpu_image_embed_cache[image_hash]
            tensors = tuple(t.to(DEVICE) for t in tensors)
            return tensors
        else:
            return None

    def compute_image_size(
        self, image: Union[np.ndarray, LazyImageRetrievalWrapper]
    ) -> Tuple[int, int]:
        if isinstance(image, LazyImageRetrievalWrapper):
            if (image_size := self.image_size_cache.get(image.image_hash)) is None:
                np_img = image.image_as_numpy
                image_size = np_img.shape[:2][::-1]
                self.image_size_cache[image.image_hash] = image_size
                image.unload_numpy_image()  # free the huge numpy array immediately
            return image_size
        else:
            return image.shape[:2][::-1]

    @torch.no_grad()
    def embed_image(self, image: Union[np.ndarray, LazyImageRetrievalWrapper]) -> Hash:
        if isinstance(image, LazyImageRetrievalWrapper):
            image_hash = image.image_hash
        else:
            image_hash = hash_function(image.tobytes())

        if (image_embeds := self.get_image_embeds(image_hash)) is not None:
            return image_hash

        np_image = (
            image.image_as_numpy
            if isinstance(image, LazyImageRetrievalWrapper)
            else image
        )
        pixel_values = preprocess_image(
            np_image, self.image_size, self.image_mean, self.image_std
        )

        # torch 2.4 lets you use "cuda:0" as device_type
        # but this crashes in 2.3
        # so we parse DEVICE as a string to make it work in both 2.3 and 2.4
        # as we don't know a priori our torch version
        device_str = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
        # we disable autocast on CPU for stability, although it's possible using bfloat16 would work
        with torch.autocast(
            device_type=device_str, dtype=torch.float16, enabled=device_str == "cuda"
        ):
            image_embeds, _ = self.model.image_embedder(pixel_values=pixel_values)
            batch_size, h, w, dim = image_embeds.shape
            image_features = image_embeds.reshape(batch_size, h * w, dim)
            objectness = self.model.objectness_predictor(image_features)
            boxes = self.model.box_predictor(image_features, feature_map=image_embeds)
        image_class_embeds = self.model.class_head.dense0(image_features)
        image_class_embeds /= (
            torch.linalg.norm(image_class_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        )
        logit_shift = self.model.class_head.logit_shift(image_features)
        logit_scale = (
            self.model.class_head.elu(self.model.class_head.logit_scale(image_features))
            + 1
        )
        objectness = objectness.sigmoid()

        objectness, boxes, image_class_embeds, logit_shift, logit_scale = (
            filter_tensors_by_objectness(
                objectness, boxes, image_class_embeds, logit_shift, logit_scale
            )
        )

        self.image_embed_cache[image_hash] = (
            objectness,
            boxes,
            image_class_embeds,
            logit_shift,
            logit_scale,
        )

        # Explicitly delete temporary tensors to free memory.
        del pixel_values, np_image, image_features, image_embeds

        if isinstance(image, LazyImageRetrievalWrapper):
            image.unload_numpy_image()  # Clears both _image_as_numpy and image if needed.

        return image_hash

    def get_query_embedding(
        self, query_spec: QuerySpecType, iou_threshold: float
    ) -> torch.Tensor:
        # NOTE: for now we're handling each image seperately
        query_embeds = []
        for image_hash, query_boxes in query_spec.items():
            image_embeds = self.get_image_embeds(image_hash)
            if image_embeds is None:
                raise KeyError("We didn't embed the image first!")
            _objectness, image_boxes, image_class_embeds, _, _ = image_embeds

            query_boxes_tensor = torch.tensor(
                query_boxes, dtype=image_boxes.dtype, device=image_boxes.device
            )
            if image_boxes.numel() == 0 or query_boxes_tensor.numel() == 0:
                continue
            iou, _ = box_iou(
                to_corners(image_boxes), to_corners(query_boxes_tensor)
            )  # 3000, k
            ious, indices = torch.max(iou, dim=0)
            # filter for only iou > 0.4
            iou_mask = ious > iou_threshold
            indices = indices[iou_mask]
            if not indices.numel() > 0:
                continue

            embeds = image_class_embeds[indices]
            query_embeds.append(embeds)
        if not query_embeds:
            return None
        query = torch.cat(query_embeds, dim=0)
        return query

    def infer_from_embed(
        self,
        image_hash: Hash,
        query_embeddings: Dict[str, PosNegDictType],
        confidence: float,
        iou_threshold: float,
    ) -> List[Dict]:
        image_embeds = self.get_image_embeds(image_hash)
        if image_embeds is None:
            raise KeyError("We didn't embed the image first!")
        _, image_boxes, image_class_embeds, _, _ = image_embeds
        class_map, class_names = make_class_map(query_embeddings)
        all_predicted_boxes, all_predicted_classes, all_predicted_scores = [], [], []
        for class_name, pos_neg_embedding_dict in query_embeddings.items():
            boxes, classes, scores = get_class_preds_from_embeds(
                pos_neg_embedding_dict,
                image_class_embeds,
                confidence,
                image_boxes,
                class_map,
                class_name,
                iou_threshold,
            )

            all_predicted_boxes.append(boxes)
            all_predicted_classes.append(classes)
            all_predicted_scores.append(scores)

        if not all_predicted_boxes:
            return []

        all_predicted_boxes = torch.cat(all_predicted_boxes, dim=0)
        all_predicted_classes = torch.cat(all_predicted_classes, dim=0)
        all_predicted_scores = torch.cat(all_predicted_scores, dim=0)

        # run nms on all predictions
        survival_indices = torchvision.ops.nms(
            to_corners(all_predicted_boxes), all_predicted_scores, iou_threshold
        )
        all_predicted_boxes = all_predicted_boxes[survival_indices]
        all_predicted_classes = all_predicted_classes[survival_indices]
        all_predicted_scores = all_predicted_scores[survival_indices]

        # move tensors to numpy before returning
        all_predicted_boxes = all_predicted_boxes.cpu().numpy()
        all_predicted_classes = all_predicted_classes.cpu().numpy()
        all_predicted_scores = all_predicted_scores.cpu().numpy()

        return [
            {
                "class_name": class_names[int(c)],
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "confidence": float(score),
            }
            for c, (x, y, w, h), score in zip(
                all_predicted_classes, all_predicted_boxes, all_predicted_scores
            )
        ]

    def infer(
        self,
        image: Any,
        training_data: Dict,
        confidence: float = 0.99,
        iou_threshold: float = 0.3,
        **kwargs,
    ):
        class_embeddings_dict = self.make_class_embeddings_dict(
            training_data, iou_threshold
        )
        return self.infer_from_embedding_dict(
            image, class_embeddings_dict, confidence, iou_threshold
        )

    def infer_from_embedding_dict(
        self,
        image: Any,
        class_embeddings_dict: Dict[str, PosNegDictType],
        confidence: float,
        iou_threshold: float,
        **kwargs,
    ):
        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        images = [LazyImageRetrievalWrapper(image) for image in images]

        results = []
        image_sizes = []
        for image_wrapper in images:
            # happy path here is that both image size and image embeddings are cached
            # in which case we avoid loading the image at all
            image_size = self.compute_image_size(image_wrapper)
            image_sizes.append(image_size)
            image_hash = self.embed_image(image_wrapper)
            image_wrapper.unload_numpy_image()
            result = self.infer_from_embed(
                image_hash, class_embeddings_dict, confidence, iou_threshold
            )
            results.append(result)
        return self.make_response(
            results, image_sizes, sorted(list(class_embeddings_dict.keys()))
        )

    def make_class_embeddings_dict(
        self,
        training_data: List[Any],
        iou_threshold: float,
        return_image_embeds: bool = False,
    ) -> Dict[str, PosNegDictType]:

        wrapped_training_data = [
            {
                "image": LazyImageRetrievalWrapper(train_image["image"]),
                "boxes": train_image["boxes"],
            }
            for train_image in training_data
        ]

        wrapped_training_data_hash = hash_wrapped_training_data(wrapped_training_data)

        if (
            class_embeddings_dict := self.class_embeddings_cache.get(
                wrapped_training_data_hash
            )
        ) is not None:
            if return_image_embeds:
                # Return a dummy empty dict as the second value
                # or extract it from CPU cache if available
                return_image_embeds_dict = {}
                for image_hash in self.cpu_image_embed_cache:
                    return_image_embeds_dict[image_hash] = self.cpu_image_embed_cache[
                        image_hash
                    ]
                return class_embeddings_dict, return_image_embeds_dict
            else:
                return class_embeddings_dict

        class_embeddings_dict = defaultdict(lambda: {"positive": [], "negative": []})

        bool_to_literal = {True: "positive", False: "negative"}
        return_image_embeds_dict = dict()

        for train_image in wrapped_training_data:
            image_hash = self.embed_image(train_image["image"])
            if return_image_embeds:
                if (image_embeds := self.get_image_embeds(image_hash)) is None:
                    raise KeyError("We didn't embed the image first!")
                return_image_embeds_dict[image_hash] = tuple(
                    t.to("cpu") for t in image_embeds
                )

            image_size = self.compute_image_size(train_image["image"])
            # grab and normalize box prompts for this image
            boxes = train_image["boxes"]
            coords = [[box["x"], box["y"], box["w"], box["h"]] for box in boxes]
            coords = [tuple([c / max(image_size) for c in coord]) for coord in coords]
            classes = [box["cls"] for box in boxes]
            is_positive = [not box["negative"] for box in boxes]
            query_spec = {image_hash: coords}
            # compute the embeddings for the box prompts
            embeddings = self.get_query_embedding(query_spec, iou_threshold)

            del train_image

            if embeddings is None:
                continue

            for embedding, class_name, is_pos in zip(embeddings, classes, is_positive):
                class_embeddings_dict[class_name][bool_to_literal[is_pos]].append(
                    embedding
                )

        gc.collect()
        # Convert lists of embeddings to tensors.
        class_embeddings_dict = {
            k: {
                "positive": torch.stack(v["positive"]) if v["positive"] else None,
                "negative": torch.stack(v["negative"]) if v["negative"] else None,
            }
            for k, v in class_embeddings_dict.items()
        }

        self.class_embeddings_cache[wrapped_training_data_hash] = class_embeddings_dict
        if return_image_embeds:
            return class_embeddings_dict, return_image_embeds_dict

        return class_embeddings_dict

    def make_response(self, predictions, image_sizes, class_names):
        responses = [
            ObjectDetectionInferenceResponse(
                predictions=[
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": pred["x"] * max(image_sizes[ind]),
                            "y": pred["y"] * max(image_sizes[ind]),
                            "width": pred["w"] * max(image_sizes[ind]),
                            "height": pred["h"] * max(image_sizes[ind]),
                            "confidence": pred["confidence"],
                            "class": pred["class_name"],
                            "class_id": class_names.index(pred["class_name"]),
                        }
                    )
                    for pred in batch_predictions
                ],
                image=InferenceResponseImage(
                    width=image_sizes[ind][0], height=image_sizes[ind][1]
                ),
            )
            for ind, batch_predictions in enumerate(predictions)
        ]
        return responses


class SerializedOwlV2(RoboflowInferenceModel):
    task_type = "object-detection"
    box_format = "xywh"

    # Cache of OwlV2 instances to avoid creating new ones for each serialize_training_data call
    # This improves performance by reusing model instances across serialization operations
    _base_owlv2_instances = {}

    @classmethod
    def get_or_create_owlv2_instance(cls, roboflow_id: str) -> OwlV2:
        """Get an existing OwlV2 instance from cache or create a new one if it doesn't exist.

        Args:
            roboflow_id: The model ID for the OwlV2 model

        Returns:
            An OwlV2 instance
        """
        if roboflow_id in cls._base_owlv2_instances:
            return cls._base_owlv2_instances[roboflow_id]
        else:
            owlv2 = OwlV2(model_id=roboflow_id)
            cls._base_owlv2_instances[roboflow_id] = owlv2
            return owlv2

    @classmethod
    def serialize_training_data(
        cls,
        training_data: List[Any],
        hf_id: str = f"google/{OWLV2_VERSION_ID}",
        iou_threshold: float = 0.3,
        save_dir: str = os.path.join(MODEL_CACHE_DIR, "owl-v2-serialized-data"),
        previous_embeddings_file: str = None,
    ):
        roboflow_id = hf_id.replace("google/", "owlv2/")

        owlv2 = cls.get_or_create_owlv2_instance(roboflow_id)

        if previous_embeddings_file is not None:
            if DEVICE == "cpu":
                model_data = torch.load(previous_embeddings_file, map_location="cpu")
            else:
                model_data = torch.load(previous_embeddings_file)

            train_data_dict = model_data["train_data_dict"]
            owlv2.cpu_image_embed_cache = model_data["image_embeds"]

        train_data_dict, image_embeds = owlv2.make_class_embeddings_dict(
            training_data, iou_threshold, return_image_embeds=True
        )
        return cls.save_model(
            hf_id, roboflow_id, train_data_dict, image_embeds, save_dir
        )

    @classmethod
    def save_model(
        cls,
        hf_id: str,
        roboflow_id: str,
        train_data_dict: Dict,
        image_embeds: Dict,
        save_dir: str,
    ):
        train_data_dict = {
            "huggingface_id": hf_id,
            "train_data_dict": train_data_dict,
            "class_names": list(train_data_dict.keys()),
            "roboflow_id": roboflow_id,
            "image_embeds": image_embeds,
        }
        train_data_path = os.path.join(save_dir, cls.weights_file_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(train_data_dict, train_data_path)
        return train_data_path

    def infer_from_request(
        self,
        request: ObjectDetectionInferenceRequest,
    ) -> Union[
        List[ObjectDetectionInferenceResponse], ObjectDetectionInferenceResponse
    ]:
        return super().infer_from_request(request)

    def __init__(self, model_id, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)
        self.get_model_artifacts()

    def get_infer_bucket_file_list(self):
        return []

    def download_model_artefacts_from_s3(self):
        raise NotImplementedError("Owlv2 not currently supported on hosted inference")

    def download_model_artifacts_from_roboflow_api(self):
        logger.info(f"Downloading OWLv2 model artifacts")
        if self.version_id is not None:
            api_data = get_roboflow_model_data(
                api_key=self.api_key,
                model_id=self.endpoint,
                endpoint_type=ModelEndpointType.OWLV2,
                device_id=self.device_id,
            )
            api_data = api_data["owlv2"]
            if "model" not in api_data:
                raise ModelArtefactError(
                    "Could not find `model` key in roboflow API model description response."
                )
            logger.info(f"Downloading OWLv2 model weights from {api_data['model']}")
            model_weights_response = get_from_url(
                api_data["model"], json_response=False
            )
        else:
            logger.info(f"Getting OWLv2 model data for")
            api_data = get_roboflow_instant_model_data(
                api_key=self.api_key,
                model_id=self.endpoint,
            )
            if (
                "modelFiles" not in api_data
                or "owlv2" not in api_data["modelFiles"]
                or "model" not in api_data["modelFiles"]["owlv2"]
            ):
                raise ModelArtefactError(
                    "Could not find `modelFiles` key or `modelFiles`.`owlv2` or `modelFiles`.`owlv2`.`model` key in roboflow API model description response."
                )
            logger.info(
                f"Downloading OWLv2 model weights from {api_data['modelFiles']['owlv2']['model']}"
            )
            model_weights_response = get_from_url(
                api_data["modelFiles"]["owlv2"]["model"], json_response=False
            )
        save_bytes_in_cache(
            content=model_weights_response.content,
            file=self.weights_file,
            model_id=self.endpoint,
        )
        logger.info(f"OWLv2 model weights saved to cache")

    def load_model_artifacts_from_cache(self):
        if DEVICE == "cpu":
            self.model_data = torch.load(
                self.cache_file(self.weights_file), map_location="cpu"
            )
        else:
            self.model_data = torch.load(self.cache_file(self.weights_file))
        self.class_names = self.model_data["class_names"]
        self.train_data_dict = self.model_data["train_data_dict"]
        self.huggingface_id = self.model_data["huggingface_id"]
        self.roboflow_id = self.model_data["roboflow_id"]
        # Use the same cached OwlV2 instance mechanism to avoid creating duplicates
        self.owlv2 = self.__class__.get_or_create_owlv2_instance(self.roboflow_id)
        self.owlv2.cpu_image_embed_cache = self.model_data["image_embeds"]

    weights_file_path = "weights.pt"

    @property
    def weights_file(self):
        return self.weights_file_path

    def infer(
        self, image, confidence: float = 0.99, iou_threshold: float = 0.3, **kwargs
    ):
        logger.info(f"Inferring OWLv2 model")
        result = self.owlv2.infer_from_embedding_dict(
            image,
            self.train_data_dict,
            confidence=confidence,
            iou_threshold=iou_threshold,
            **kwargs,
        )
        logger.info(f"OWLv2 model inference complete")
        return result

    def draw_predictions(
        self,
        inference_request: ObjectDetectionInferenceRequest,
        inference_response: ObjectDetectionInferenceResponse,
    ):
        return self.owlv2.draw_predictions(
            inference_request,
            inference_response,
        )

    def save_small_model_without_image_embeds(
        self, save_dir: str = os.path.join(MODEL_CACHE_DIR, "owl-v2-serialized-data")
    ):
        self.owlv2.cpu_image_embed_cache = LimitedSizeDict(
            size_limit=CPU_IMAGE_EMBED_CACHE_SIZE
        )
        return self.save_model(
            self.huggingface_id,
            self.roboflow_id,
            self.train_data_dict,
            self.owlv2.cpu_image_embed_cache,
            save_dir,
        )
