# Roboflow API Key
Throughout these docs you will see references to your Roboflow API key. Using your Roboflow API key grants you access to the models you have trained on Roboflow, public models avialable on Roboflow univers, and access to hosted inference API's.

## Access Your Roboflow API Key

For every example in the following documentation you will need to provide your Roboflow API key. To access your Roboflow API key, you will need to [create a free Roboflow account](https://app.roboflow.com), then [follow the docs](https://docs.roboflow.com/api-reference/authentication) to retrieve your key.

## Use Your Roboflow API Key

How you use your Roboflow API key will depend on how you are using `inference`.
### Python SDK
 Within the python SDK, your Roboflow API key can be set via keyword arguments

```python
from inference.models.utils import get_roboflow_model

model = get_roboflow_model(model_id="...", api_key="YOUR ROBOFLOW API KEY")

from inference.models import YOLOv8ObjectDetection

model = YOLOv8ObjectDetection(model_id="...", api_key="YOUR ROBOFLOW API KEY")
```

or it can be set via environment variables and the keyword arguments can be omitted

```bash
export ROBOFLOW_API_KEY=<YOUR ROBOFLOW API KEY>
```

```python
from inference.models.utils import get_roboflow_model

model = get_roboflow_model(model_id="...")
 ```

### HTTP Request
When using HTTP requests, your Roboflow API key should be passed as a url parameter, or as part of the request payload, depending on the route you are using.
```python
import requests

my_api_key = "YOUR ROBOFLOW API KEY"

url = f"http://localhost:9001/soccer-players-5fuqs/1?api_key={my_api_key}"
response = requests.post(url,...)

url = "http://localhost:9001/infer/object_detection"
payload = {
  api_key: my_api_key,
  model_id: "soccer-players-5fuqs/1",
  ...
}
response = requests.post(url,json=payload)
```

### Docker Configuration
If you are running the Roboflow Inference Server in a docker container, you can provide your Roboflow API key within the `docker run` command.

```bash
docker run -it --rm --network=host -e ROBOFLOW_API_KEY=YOUR_ROBOFLOW_API_KEY roboflow/roboflow-inference-server-cpu:latest
```