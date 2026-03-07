import os
import re
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
    endpoint = "moondream2/moondream2_2b_jul24"
    trust_remote_code = True
    revision = "2025-06-21"

    def __init__(self, *args, **kwargs):
        # if model_id in kwargs, delete
        if "model_id" in kwargs:
            del kwargs["model_id"]

        super().__init__(self.endpoint, *args, **kwargs)

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        if self.endpoint == "moondream2/moondream2_2b_jul24":
            return [
                "added_tokens.json",
                "config.json",
                "config.py",
                "configuration_moondream.py",
                "fourier_features.py",
                "generation_config.json",
                "handler.py",
                "hf_moondream.py",
                "image_crops.py",
                "layers.py",
                "lora.py",
                re.compile(r"model.*\.safetensors"),
                "moondream.py",
                "region.py",
                "region_model.py",
                "rope.py",
                "special_tokens_map.json",
                "text.py",
                "tokenizer.json",
                "tokenizer_config.json",
                "utils.py",
                "versions.txt",
                "vision.py",
                "vision_encoder.py",
                "vocab.json",
                "weights.py",
            ]
        else:
            return super().get_infer_bucket_file_list()

    def initialize_model(self, **kwargs):
        model_cls = import_class_from_file(
            os.path.join(self.cache_dir, "hf_moondream.py"),
            "HfMoondream",
        )

        # The downloaded HfMoondream doesn't call self.post_init() at the end
        # of __init__, which transformers 5.x requires to set
        # all_tied_weights_keys and other attributes.
        _original_init = model_cls.__init__

        def _patched_init(self_inner, config):
            _original_init(self_inner, config)
            self_inner.post_init()

        model_cls.__init__ = _patched_init

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model_cls.from_pretrained(self.cache_dir).to(device)
        # Recompute non-persistent buffers (freqs_cis and attn_mask) after
        # loading.  These are registered with persistent=False and are not
        # saved in the checkpoint.  transformers 5.x's from_pretrained/
        # post_init resets them to uninitialized memory.
        config = self.model.model.config
        tc = config.text

        precompute_freqs_cis = import_class_from_file(
            os.path.join(self.cache_dir, "rope.py"),
            "precompute_freqs_cis",
        )
        self.model.model.text.freqs_cis = precompute_freqs_cis(
            tc.dim // (2 * tc.n_heads), tc.max_context
        ).to(device)

        attn_mask = torch.tril(
            torch.ones(1, 1, tc.max_context, tc.max_context, dtype=torch.bool)
        )
        patch_w = config.vision.crop_size // config.vision.enc_patch_size
        prefix_attn_len = 1 + patch_w**2
        attn_mask[..., :prefix_attn_len, :prefix_attn_len] = 1
        self.model.model.attn_mask = attn_mask.to(device)

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
                            "class": prompt if prompt is not None else "",
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
