from .base import BaseOCRModel
from inference.core.workflows.core_steps.common.entities import (
    StepExecutionMode,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from typing import Optional, List, Callable
from inference.core.workflows.prototypes.block import BlockResult

import requests
import json
import base64


class MathpixOCRModel(BaseOCRModel):
    def __init__(
        self,
        model_manager,
        api_key: Optional[str],
        mathpix_app_id: str,
        mathpix_app_key: str,
    ):
        super().__init__(model_manager, api_key)
        self.mathpix_app_id = mathpix_app_id
        self.mathpix_app_key = mathpix_app_key

    def run(
        self,
        images: Batch[WorkflowImageData],
        step_execution_mode: StepExecutionMode,
        post_process_result: Callable[
            [Batch[WorkflowImageData], List[dict]], BlockResult
        ],
    ) -> BlockResult:
        predictions = []
        for image_data in images:
            # Decode base64 image to bytes
            image_bytes = base64.b64decode(image_data.base64_image)

            # Prepare the request
            url = "https://api.mathpix.com/v3/text"
            headers = {
                "app_id": self.mathpix_app_id,
                "app_key": self.mathpix_app_key,
            }
            data = {
                "options_json": json.dumps(
                    {
                        "math_inline_delimiters": ["$", "$"],
                        "rm_spaces": True,
                    }
                )
            }
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

            # Send the request
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files=files,
            )

            if response.status_code == 200:
                result = response.json()
                # Extract the text result
                text = result.get("text", "")
            else:
                error_info = response.json().get("error", {})
                message = error_info.get("message", response.text)
                detailed_message = error_info.get("detail", "")

                raise Exception(
                    f"Mathpix API request failed: {message} \n\n"
                    f"Detailed: {detailed_message}"
                )

            prediction = {"result": text}
            predictions.append(prediction)

        return post_process_result(images, predictions)
