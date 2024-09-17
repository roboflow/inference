import hashlib
import os
from collections import defaultdict
from typing import Dict, List, NewType

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.models.owlv2.modeling_owlv2 import box_iou

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import DEVICE
from inference.core.models.roboflow import (
    DEFAULT_COLOR_PALETTE,
    RoboflowCoreModel,
    draw_detection_predictions,
)
from inference.core.utils.image_utils import load_image_rgb

Hash = NewType("Hash", str)
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


class OwlV2(RoboflowCoreModel):
    task_type = "object-detection"
    box_format = "xywh"

    def __init__(self, *args, model_id="owlv2/owlv2-base-patch16-ensemble", **kwargs):
        super().__init__(*args, model_id=model_id, **kwargs)
        hf_id = os.path.join("google", self.version_id)
        self.processor = Owlv2Processor.from_pretrained(hf_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(hf_id).eval().to(DEVICE)
        self.image_embed_cache = LimitedSizeDict(
            size_limit=50
        )  # NOTE: this should have a max size

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

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> Hash:
        image_hash = hashlib.sha256(np.array(image).tobytes()).hexdigest()

        if (image_embeds := self.image_embed_cache.get(image_hash)) is not None:
            return image_hash

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(DEVICE)
        image_embeds, _ = self.model.image_embedder(pixel_values=pixel_values)
        batch_size, h, w, dim = image_embeds.shape
        image_features = image_embeds.reshape(batch_size, h * w, dim)
        objectness = self.model.objectness_predictor(image_features)
        boxes = self.model.box_predictor(image_features, feature_map=image_embeds)

        # class_embeddings =  model.class_predictor(image_features)[1]
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

        self.image_embed_cache[image_hash] = (
            objectness.squeeze(0),
            boxes.squeeze(0),
            image_class_embeds.squeeze(0),
            logit_shift.squeeze(0).squeeze(1),
            logit_scale.squeeze(0).squeeze(1),
        )

        return image_hash

    def get_query_embedding(self, query_spec: Dict[Hash, List[List[int]]]):
        # NOTE: for now we're handling each image seperately
        query_embeds = []
        for image_hash, query_boxes in query_spec.items():
            try:
                objectness, image_boxes, image_class_embeds, _, _ = (
                    self.image_embed_cache[image_hash]
                )
            except KeyError as error:
                raise KeyError("We didn't embed the image first!") from error

            query_boxes_tensor = torch.tensor(
                query_boxes, dtype=torch.float, device=image_boxes.device
            )
            iou, union = box_iou(
                to_corners(image_boxes), to_corners(query_boxes_tensor)
            )  # 3000, k
            iou_mask = iou > 0.4
            valid_objectness = torch.where(
                iou_mask, objectness.unsqueeze(-1), -1
            )  # 3000, k
            if torch.all(iou_mask == 0):
                continue
            else:
                indices = torch.argmax(valid_objectness, dim=0)
                embeds = image_class_embeds[indices]
                query_embeds.append(embeds)
        if not query_embeds:
            return None
        query = torch.cat(query_embeds).mean(dim=0)
        query /= torch.linalg.norm(query, ord=2) + 1e-6
        return query

    def infer_from_embed(self, image_hash: Hash, query_embeddings, confidence):
        objectness, image_boxes, image_class_embeds, logit_shift, logit_scale = (
            self.image_embed_cache[image_hash]
        )
        predicted_boxes = []
        predicted_classes = []
        predicted_scores = []
        class_names = sorted(list(query_embeddings.keys()))
        class_map = {class_name: i for i, class_name in enumerate(class_names)}
        for class_name, embedding in query_embeddings.items():
            if embedding is None:
                continue
            pred_logits = torch.einsum("sd,d->s", image_class_embeds, embedding)
            pred_logits = (pred_logits + logit_shift) * logit_scale
            prediction_scores = pred_logits.sigmoid()
            score_mask = prediction_scores > confidence
            predicted_boxes.append(image_boxes[score_mask, :])
            scores = prediction_scores[score_mask]
            predicted_scores.append(scores)
            class_ind = class_map[class_name]
            predicted_classes.append(class_ind * torch.ones_like(scores))

        all_boxes = torch.cat(predicted_boxes, dim=0)
        all_classes = torch.cat(predicted_classes, dim=0)
        all_scores = torch.cat(predicted_scores, dim=0)
        survival_indices = torchvision.ops.nms(to_corners(all_boxes), all_scores, 0.3)
        pred_boxes = all_boxes[survival_indices].detach().cpu().numpy()
        pred_classes = all_classes[survival_indices].detach().cpu().numpy()
        pred_scores = all_scores[survival_indices].detach().cpu().numpy()
        return [
            {
                "class_name": class_names[int(c)],
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "confidence": float(score),
            }
            for c, (x, y, w, h), score in zip(pred_classes, pred_boxes, pred_scores)
        ]

    def infer(self, image, training_data, confidence=0.99, **kwargs):
        class_to_query_spec = defaultdict(lambda: defaultdict(list))
        for train_image_dict in training_data:
            boxes, train_image = train_image_dict["boxes"], train_image_dict["image"]
            train_image = load_image_rgb(train_image)
            image_hash = self.embed_image(train_image)
            for box in boxes:
                class_name = box["cls"]
                coords = box["x"], box["y"], box["w"], box["h"]
                coords = tuple([c / max(train_image.shape[:2]) for c in coords])
                class_to_query_spec[class_name][image_hash].append(coords)

        my_class_to_embeddings_dict = dict()
        for class_name, query_spec in class_to_query_spec.items():
            class_embedding = self.get_query_embedding(query_spec)
            my_class_to_embeddings_dict[class_name] = class_embedding

        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        results = []
        image_sizes = []
        for image in images:
            image = load_image_rgb(image)
            image_sizes.append(image.shape[:2][::-1])
            image_hash = self.embed_image(image)
            result = self.infer_from_embed(
                image_hash, my_class_to_embeddings_dict, confidence
            )
            results.append(result)
        return self.make_response(
            results, image_sizes, sorted(list(my_class_to_embeddings_dict.keys()))
        )

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
