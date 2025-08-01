# Find Your Roboflow API Key

Using your Roboflow API key grants you access to the models you have trained on Roboflow, public models available on Roboflow Universe, and access to hosted inference APIs (i.e. the [Roboflow Serverless API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2)).

[Read the Roboflow product documentation to learn how find your API key.](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key)

You will need an API key for everything listed in the table below that does not have a check in the "Open Access" column:

|                         | Open Access | With API Key |
|-------------------------|-------------|--------------|
| [Pre-Trained Models](https://inference.roboflow.com/quickstart/aliases/#supported-pre-trained-models) | ✅ | ✅
| [Foundation Models](https://inference.roboflow.com/foundation/about/) | ✅ | ✅
| [Video Stream Management](https://inference.roboflow.com/workflows/video_processing/overview/) | ✅ | ✅
| [Dynamic Python Blocks](https://inference.roboflow.com/workflows/custom_python_code_blocks/) | ✅ | ✅
| [Public Workflows](https://inference.roboflow.com/workflows/about/) | ✅ | ✅
| [Private Workflows](https://docs.roboflow.com/workflows/create-a-workflow) |  | ✅
| [Fine-Tuned Models](https://roboflow.com/train) |  | ✅
| [Universe Models](https://roboflow.com/universe) |  | ✅
| [Active Learning](https://inference.roboflow.com/workflows/blocks/roboflow_dataset_upload/) |  | ✅
| [Serverless Hosted API](https://docs.roboflow.com/deploy/hosted-api) |  | ✅
| [Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments) |  | ✅
| [Commercial Model Licensing](https://roboflow.com/licensing) |  | Paid
| [Device Management](https://docs.roboflow.com/roboflow-enterprise) |  | Enterprise
| [Model Monitoring](https://docs.roboflow.com/deploy/model-monitoring) |  | Enterprise

## Use Your Roboflow API Key

There are several ways to configure your Roboflow API key when using Inference.

### In a Workflow Call

When you run a Roboflow Workflow, you will need to specify your API key.

#### With the Inference SDK (for Images)

If you are using the Inference SDK, you can set up your API key with your HTTP client:

```python
from inference_sdk import InferenceHTTPClient
import os

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")
)
```

#### With the InferencePipeline Method (for Video)

If you are running your Workflow on a video, you will use InferencePipeline. You can set your API key using the `api_key` argument:

```python
from inference import InferencePipeline
import os

pipeline = InferencePipeline.init_with_workflow(
    api_key=os.environ.get("ROBOFLOW_API_KEY")
    workspace_name="workspace-name",
    workflow_id="custom-workflow-84",
    video_reference=0,
    max_fps=30
)
```

You can find this code snippet from the "Deploy" section of the Roboflow Workflows web interface.

### Environment Variable

You can also set your Roboflow API key within your environment via the variable `ROBOFLOW_API_KEY`. In most terminals you can run:

```bash
export ROBOFLOW_API_KEY=MY_ROBOFLOW_API_KEY
```

This is useful if you are using the Inference Python package directly.

Then, any command you run within that same terminal session will have access to the environment variable `ROBOFLOW_API_KEY`.

<details>
<summary>Advanced</summary>
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
</details>