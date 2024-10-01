# models/google_cloud_vision.py

from .base import BaseOCRModel
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from typing import Optional, List
import requests
import base64


class GoogleCloudVisionOCRModel(BaseOCRModel):
    def __init__(
        self, model_manager, api_key: Optional[str], google_cloud_api_key: str
    ):
        super().__init__(model_manager, api_key)
        self.google_cloud_api_key = google_cloud_api_key

    def run(
        self,
        images: Batch[WorkflowImageData],
        step_execution_mode: StepExecutionMode,
        post_process_result,
    ):
        predictions = []
        for image_data in images:
            # Use base64_image directly
            encoded_image = image_data.base64_image
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_cloud_api_key}"

            payload = {
                "requests": [
                    {
                        "image": {"content": encoded_image},
                        "features": [{"type": "TEXT_DETECTION"}],
                    }
                ]
            }
            # Send the request
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                text_annotations = result["responses"][0].get("textAnnotations", [])
                if text_annotations:
                    text = text_annotations[0]["description"]
                else:
                    text = ""
            else:
                error_info = response.json().get("error", {})
                message = error_info.get("message", response.text)
                raise Exception(f"Google Cloud Vision API request failed: {message}")
            prediction = {"result": text}
            predictions.append(prediction)
        return post_process_result(images, predictions)
