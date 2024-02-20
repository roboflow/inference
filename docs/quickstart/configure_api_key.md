# Roboflow API Key

Throughout these docs you will see references to your Roboflow API key. Using your Roboflow API key grants you access to the models you have trained on Roboflow, public models available on Roboflow Universe, and access to hosted inference API's.

## Access Your Roboflow API Key

For some examples in the documentation you will need to provide your Roboflow API key. To access your Roboflow API key, you will need to <a href="https://app.roboflow.com" target="_blank">create a free Roboflow account</a>, then <a href="https://docs.roboflow.com/api-reference/authentication" target="_blank">follow the docs</a> to retrieve your key.

## Use Your Roboflow API Key

There are several ways to configure your Roboflow API key when using Inference.

### Environment Variable

The recommended way is to set your Roboflow API key within your environment via the variable `ROBOFLOW_API_KEY`. In most terminals you can run:

```bash
export ROBOFLOW_API_KEY=MY_ROBOFLOW_API_KEY
```

Then, any command you run within that same terminal session will have access to the environment variable `ROBOFLOW_API_KEY`.

### Python

When using Inference within python, your Roboflow API key can be set via keyword arguments

```python
from inference.models.utils import get_model

model = get_model(model_id="...", api_key="YOUR ROBOFLOW API KEY")
```

!!! Hint

    If you set your API key in your environment, you do not have to pass it as a keyword argument: `model = get_model(model_id="...")`

### HTTP Request Payload

When using HTTP requests, your Roboflow API key should be passed as a url parameter, or as part of the request payload, depending on the route you are using.

```python
import requests

my_api_key = "YOUR ROBOFLOW API KEY"

url = f"http://localhost:9001/soccer-players-5fuqs/1?api_key={my_api_key}"
response = requests.post(url,...)

url = "http://localhost:9001/infer/object_detection"
payload = {
  "api_key": my_api_key,
  "model_id": "soccer-players-5fuqs/1",
  ...
}
response = requests.post(url,json=payload)
```

### Docker Configuration

If you are running the Roboflow Inference Server locally in a docker container, you can provide your Roboflow API key within the `docker run` command.

```bash
docker run -it --rm --network=host -e ROBOFLOW_API_KEY=YOUR_ROBOFLOW_API_KEY roboflow/roboflow-inference-server-cpu:latest
```

Requests sent to this server can now omit `api_key` from the request payload.
