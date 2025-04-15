import os
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import MODEL_CACHE_DIR
from inference.core.models.base import PreprocessReturnMetadata
from inference.models.florence2.utils import import_class_from_file
from inference.models.transformers import TransformerModel


class Moondream2(TransformerModel):
    generation_includes_input = True
    transformers_class = AutoModelForCausalLM
    load_base_from_roboflow = True
    version_id = None
    default_dtype = torch.bfloat16
    load_weights_as_transformers = True
    endpoint = "moondream2/moondream2-2b"
    trust_remote_code = True
    revision = "2025-03-27"

    def __init__(self, *args, **kwargs):
        # if model_id in kwargs, delete
        if "model_id" in kwargs:
            del kwargs["model_id"]

        super().__init__(self.endpoint, *args, **kwargs)

    def initialize_model(self):
        model = import_class_from_file(
            os.path.join(self.cache_dir, "hf_moondream.py"),
            "HfMoondream",
        )

        self.model = model.from_pretrained(self.cache_dir)

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        return self.detect(image_in, prompt=prompt, history=history, **kwargs)

    def caption(self, image_in: Image.Image, history=None, **kwargs):
        image_in = self.model.encode_image(image_in)
        return self.model.caption(image_in, length="normal")["caption"]

    def query(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        image_in = self.model.encode_image(image_in)
        return self.model.query(image_in, prompt)["answer"]

    def detect(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        image = self.model.encode_image(image=image_in)
        return self.make_response(
            [self.model.detect(image, prompt)["objects"]],
            [image_in.size],
            prompt=prompt,
        )

    def make_response(self, predictions, image_sizes, prompt=""):
        responses = []

        for ind, batch_predictions in enumerate(predictions):
            predictions = []
            for pred in batch_predictions:
                x_min = abs(pred["x_min"]) * image_sizes[ind][0]
                y_min = abs(pred["y_min"]) * image_sizes[ind][1]
                x_max = abs(pred["x_max"]) * image_sizes[ind][0]
                y_max = abs(pred["y_max"]) * image_sizes[ind][1]
                # convert to xywh
                width = (abs(pred["x_max"]) - abs(pred["x_min"])) * image_sizes[ind][0]
                height = (abs(pred["y_max"]) - abs(pred["y_min"])) * image_sizes[ind][1]
                x = (x_min + x_max) / 2
                y = (y_min + y_max) / 2
                predictions.append(
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "confidence": 1.0,  # confidence is not returned by the model
                            "class": prompt,
                            "class_id": 0,  # you can only prompt for one object at once
                        }
                    )
                )

            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(
                        width=image_sizes[ind][0], height=image_sizes[ind][1]
                    ),
                )
            )
        return responses

    def postprocess(
        self,
        predictions: Tuple[str],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        return predictions
