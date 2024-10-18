import os

import torch
import requests

from typing import Any, Tuple

from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from inference.core.models.base import PreprocessReturnMetadata
from inference.models.transformers.transformers import TransformerModel
from inference.core.utils.image_utils import load_image_rgb

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class RTDETR(TransformerModel):
    def __init__(self, *args, model_id=f"", **kwargs):
        super().__init__(*args, model_id=model_id, **kwargs)
        self.model_id = model_id
        self.endpoint = model_id
        self.api_key = API_KEY
        self.dataset_id, self.version_id = model_id.split("/")
        self.cache_dir = os.path.join(MODEL_CACHE_DIR, self.endpoint + "/") # "PekingU/rtdetr_r50vd"
        dtype = torch.bfloat16
        self.model = RTDetrForObjectDetection.from_pretrained(
            self.cache_dir,
            torch_dtype=dtype,
            device_map=DEVICE,
            revision="bfloat16",
        ).eval()

        self.processor = RTDetrImageProcessor.from_pretrained(
            self.cache_dir,
        )
        # self.task_type = "lmm"

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[Image.Image, PreprocessReturnMetadata]:
        pil_image = Image.fromarray(load_image_rgb(image))

        return pil_image, PreprocessReturnMetadata({})

    def postprocess(
        self,
        predictions: Tuple[str],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions[0]
    
    def predict(self, image_in: Image.Image, **kwargs):
        model_inputs = self.processor(
            images=image_in, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**model_inputs)
            
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image_in.size[::-1]]), threshold=0.3)

        return results
    
if __name__ == "__main__":
    m = RTDETR()
    print(m.infer())

