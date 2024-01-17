You can use Inference without a trained model. This is useful for testing your integration with Inference, or for collecting data for Active Learning.

We call model endpoints without a trained model "stubs".

## Create a Stub

To create a stub, first create workspace and project on the <a href="https://app.roboflow.com" target="_blank">Roboflow platform</a>. Then, use the Inference
client to send request to your Inference HTTP API:

```python
import cv2
from inference_sdk import InferenceHTTPClient
import os

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=os.environ["API_KEY"],
)

image = cv2.imread(...)
CLIENT.infer(image, model_id="YOUR-PROJECT-NAME/0")
```

Use `/0` at the end of your model ID to denote you want to use the stub model endpoint.

You will receive a response in the following format:

```json
{
    "time": 0.0002442499971948564,
    "is_stub": true,
    "model_id": "asl-poly-instance-seg/0",
    "task_type": "instance-segmentation"
}
```