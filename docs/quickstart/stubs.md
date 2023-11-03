## `inference` without model? Is that possible?
Inference offers a way to expose models stub - which will not produce any meaningful predictions, but can be used for
several purposes:
* initial integration on your end with `inference` serving
* collecting dataset via `inference` Active Learning capabilities

## How stubs work?
Simply, create workspace and project at [Roboflow platform](https://app.roboflow.com). Once you are done - use the 
client to send request to the API:

```python
import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # if inference docker container is running locally
    api_key="XXX"
)

image = cv2.imread(...)
CLIENT.infer(image, model_id="YOUR-PROJECT-NAME/0")   # use version "0" to denote that you want stub model
```

As a result - you will receive the following response:
```json
{
    "time": 0.0002442499971948564,
    "is_stub": true,
    "model_id": "asl-poly-instance-seg/0",
    "task_type": "instance-segmentation"
}
```

You should not rely on response format, as it will change once you train and deploy a model, but utilising stubs
let you avoid integration cold start.
