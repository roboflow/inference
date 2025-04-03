import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from inference.models.transformers import TransformerModel
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)

class Moondream2(TransformerModel):
    generation_includes_input = True
    transformers_class = AutoModelForCausalLM
    load_base_from_roboflow = True
    version_id = None
    default_dtype = torch.bfloat16
    load_weights_as_transformers = True
    endpoint = "moondream2/moondream2"
    trust_remote_code = True
    revision = "2025-03-27"

    def __init__(self, *args, **kwargs):
        if not "model_id" in kwargs:
            kwargs["model_id"] = self.endpoint
        print("Loading model from:", self.endpoint)
        super().__init__(*args, **kwargs)

    def caption(self, image_in: Image.Image, history=None, **kwargs):
        return self.model.caption(image_in, length="normal", stream=True)["caption"]
    
    def query(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        return self.model.query(image_in, prompt)["answer"]
    
    def detect(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        return self.make_response(self.model.detect(image_in, prompt)["objects"], 
                             image_in.size)
    
    def make_response(self, predictions, image_sizes):
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
