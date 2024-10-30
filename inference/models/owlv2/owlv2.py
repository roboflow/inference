import hashlib
import os
from collections import defaultdict
from typing import Dict, List, NewType

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
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


def preprocess_image(np_image: np.ndarray, image_size: tuple[int, int], image_mean: torch.Tensor, image_std: torch.Tensor) -> torch.Tensor:
    current_size = np_image.shape[:2]

    r = min(image_size[0] / current_size[0], image_size[1] / current_size[1])
    target_size = (int(r * current_size[0]), int(r * current_size[1]))

    torch_image = torch.tensor(np_image).permute(2, 0, 1).unsqueeze(0).to(DEVICE).to(dtype=torch.float32) / 255.0
    torch_image = F.interpolate(torch_image, size=target_size, mode="bilinear", align_corners=False)

    padded_image_tensor = torch.ones((1, 3, *image_size), device=DEVICE) * 0.5
    padded_image_tensor[:, :, : torch_image.shape[2], : torch_image.shape[3]] = (
        torch_image
    )

    padded_image_tensor = (padded_image_tensor - image_mean) / image_std

    return padded_image_tensor


class OwlV2(RoboflowCoreModel):
    task_type = "object-detection"
    box_format = "xywh"

    def __init__(self, *args, model_id="owlv2/owlv2-large-patch14-ensemble", **kwargs):
        super().__init__(*args, model_id=model_id, **kwargs)
        hf_id = os.path.join("google", self.version_id)
        processor = Owlv2Processor.from_pretrained(hf_id)
        self.image_size = tuple(processor.image_processor.size.values())
        self.image_mean = torch.tensor(processor.image_processor.image_mean, device=DEVICE).view(1, 3, 1, 1)
        self.image_std = torch.tensor(processor.image_processor.image_std, device=DEVICE).view(1, 3, 1, 1)
        self.model = Owlv2ForObjectDetection.from_pretrained(hf_id).eval().to(DEVICE)
        self.reset_cache()
        
        # compile the model
        # NOTE that this is able to fix the manual attention implementation used in OWLv2
        # so we don't have to force in flash attention by ourselves
        # however that is only true if torch version 2.4 or later is used
        # for torch < 2.4, this is a LOT slower and using flash attention by ourselves is faster
        # this also breaks in torch < 2.1 so we supress torch._dynamo errors
        torch._dynamo.config.suppress_errors = True
        self.model.owlv2.vision_model = torch.compile(
            self.model.owlv2.vision_model
        )

    def reset_cache(self):
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
    def embed_image(self, image: np.ndarray) -> Hash:
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()

        if (image_embeds := self.image_embed_cache.get(image_hash)) is not None:
            return image_hash

        pixel_values = preprocess_image(image, self.image_size, self.image_mean, self.image_std)

        # torch 2.4 lets you use "cuda:0" as device_type
        # but this crashes in 2.3
        # so we parse DEVICE as a string to make it work in both 2.3 and 2.4
        # as we don't know a priori our torch version
        device = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # we use bfloat16 to support both CPU and GPU
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

        objectness, objectness_indices = torch.topk(objectness, int(0.1 * objectness.size(0)), dim=0)

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
                query_boxes, dtype=image_boxes.dtype, device=image_boxes.device
            )
            print("QUERY SHAPE")
            print(query_boxes_tensor.shape)
            iou, union = box_iou(
                to_corners(image_boxes), to_corners(query_boxes_tensor)
            )  # 3000, k
            print("iou first", iou.shape)
            iou_mask = iou > 0.4
            valid_objectness = torch.where(
                iou_mask, objectness.unsqueeze(-1), -1
            )  # 3000, k
            print("valid_objectness_shape", objectness.shape)
            objectness_values, objectness_indices = torch.topk(objectness, int(0.1 * objectness.size(0)), dim=0)
            iou = iou[objectness_indices]
            print("iou", iou.shape)
            indices = torch.argmax(iou, dim=0)
            print("indices shape", indices.shape)
            print("indices", indices)
            print("objectness_indices", objectness_indices, "indices", indices)
            print("objectness_indices shape", objectness_indices.shape,"indices shape",  indices.shape)
            print("iou_shape", iou.shape)
            valid_image_embeds = image_class_embeds[objectness_indices]
            embeds = valid_image_embeds[indices]
            print(embeds.shape)
            query_embeds.append(embeds)
        if not query_embeds:
            return None
        query = torch.cat(query_embeds, dim=0)
        query /= torch.linalg.norm(query, ord=2) + 1e-6
        print(query.shape)
        return query

    def infer_from_embed(self, image_hash: Hash, query_embeddings, confidence):
        objectness, image_boxes, image_class_embeds, logit_shift, logit_scale = (
            self.image_embed_cache[image_hash]
        )
        predicted_boxes = []
        predicted_classes = []
        predicted_scores = []
        class_names = sorted(list(query_embeddings.keys()))
        class_map = {
            (class_name, "positive"): i for i, class_name in enumerate(class_names)
        }
        class_map = {
            **class_map,
            **{
                (class_name, "negative"): i + len(class_names)
                for i, class_name in enumerate(class_names)
            },
        }
        all_boxes, all_classes, all_scores = [], [], []
        for class_name, pos_neg_embedding_dict in query_embeddings.items():
            predicted_boxes = []
            predicted_classes = []
            predicted_scores = []
            positive_arr = []
            for positive, embedding in pos_neg_embedding_dict.items():
                print((positive, embedding))
                if embedding is None:
                    continue
                print(image_class_embeds[:10])
                pred_logits = torch.einsum("sd,nd->ns", image_class_embeds, embedding)
                prediction_scores = pred_logits.max(dim=0)[0]
                prediction_scores = (prediction_scores + 1) / 2
                print(prediction_scores.min(), prediction_scores.max())
                score_mask = prediction_scores > confidence
                predicted_boxes.append(image_boxes[score_mask])
                scores = prediction_scores[score_mask]
                print(scores)
                predicted_scores.append(scores)
                class_ind = class_map[(class_name, positive)]
                predicted_classes.append(class_ind * torch.ones_like(scores))
                positive_arr.append(int(positive == "positive") * torch.ones_like(scores))
            

            pred_boxes = torch.cat(predicted_boxes, dim=0).float()
            pred_classes = torch.cat(predicted_classes, dim=0).float()
            pred_scores = torch.cat(predicted_scores, dim=0).float()
            positive = torch.cat(positive_arr, dim=0).float()
            survival_indices = torchvision.ops.nms(
                to_corners(pred_boxes), pred_scores, 0.3
            )
            pred_boxes = pred_boxes[survival_indices, :].detach().cpu().numpy()
            pred_classes = pred_classes[survival_indices].detach().cpu().numpy()
            pred_scores = pred_scores[survival_indices].detach().cpu().numpy()
            positive = positive[survival_indices].detach().cpu().numpy()
            is_positive = positive == 1
            all_boxes.extend(pred_boxes[is_positive])
            all_classes.extend(pred_classes[is_positive])
            all_scores.extend(pred_scores[is_positive])
        return [
            {
                "class_name": class_names[int(c)],
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "confidence": float(score),
            }
            for c, (x, y, w, h), score in zip(all_classes, all_boxes, all_scores)
        ]

    def infer(self, image, training_data, confidence=0.99, **kwargs):
        class_to_query_spec = defaultdict(lambda: defaultdict(list))
        for train_image_dict in training_data:
            boxes, train_image = train_image_dict["boxes"], train_image_dict["image"]
            train_image = load_image_rgb(train_image)
            image_hash = self.embed_image(train_image)
            for box in boxes:
                negative = box["negative"]
                positive = not negative
                class_name = box["cls"]
                coords = box["x"], box["y"], box["w"], box["h"]
                coords = tuple([c / max(train_image.shape[:2]) for c in coords])
                class_to_query_spec[(class_name, positive)][image_hash].append(coords)

        my_class_to_embeddings_dict = defaultdict(
            lambda: {"positive": None, "negative": None}
        )
        class_pos = {True: "positive", False: "negative"}
        for (class_name, positive), query_spec in class_to_query_spec.items():
            class_embedding = self.get_query_embedding(query_spec)
            my_class_to_embeddings_dict[class_name][
                class_pos[positive]
            ] = class_embedding

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
