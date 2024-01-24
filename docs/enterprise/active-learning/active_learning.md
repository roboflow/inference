# Active Learning

You can collect images automatically from Inference deployments for use in training new versions of a model. This is called active learning.

By collecting data actively that you can later label and use in training, you can gather images representative of the 
environment in which your model is deployed. These images can be used to train new model versions.

Here is the standard workflow for active learning:

* The user initiates the creation of an Active Learning configuration within the Roboflow app.
* This configuration is then distributed across all active inference instances, which may include those running against video streams and the HTTP API, both on-premises and within the Roboflow platform.
* During runtime, as predictions are generated, images and model predictions (treated as initial annotations) are dynamically collected and submitted in batches into user project. These batches are then ready for labeling within the Roboflow platform.

How active learning works with Inference is configured in your server active learning configuration. [Learn how to configure active learning](#active-learning-configuration).

Active learning can be disabled by setting `ACTIVE_LEARNING_ENABLED=false` in the environment where you run `inference`.

## Sampling Strategies

Inference supports five strategies for sampling image data for use in training new model versions. These strategies are:

* [Random sampling](#random-sampling): Images are collected at random.
* [Close-to-threshold](#close-to-threshold-sampling): Collect data close to a given threshold.
* [Detection count-based](#detections-number-based-sampling) (Detection models only): Collect data with a specific number of detections returned by a detection model.
* [Class-based](#classes-based-sampling) (Classification models only): Collect data with a specific class returned by a classification model.

You can specify multiple sampling strategies in your active learning configuration.

## How Active Learning Works

When you run Inference with an active learning configuration, the following steps are run:

1. Sampling methods are evaluated to decide which ones are applicable to the image and prediction (evaluation happens in the order of definition in your configuration).
2. A global limit for batch (defined in `batching_strategy`) is checked. Its violation terminates Active Learning attempt.
* Matching methods are checked against limits defined within their configurations. The first method with matching limit is selected.

Once a datapoint is selected and there is no limit violation, it will be saved into Roboflow platform with tags relevant for specific strategy (and global tags defined at the level of Active Learning configuration).

## Active Learning Configuration

Active learning configuration contains sampling strategies and other fields influencing the feature behaviour.

Active learning configurations are saved on the Roboflow platform and downloaded by your Inference server for use.

#### Configuration options

* `enabled`: boolean flag to enable / disable the configuration (required) - `{"enabled": false}` is minimal valid config
* `max_image_size`: two element list with positive integers (height, width) enforcing down-sizing (with aspect-ratio
preservation) of images before submission into Roboflow platform (optional)
* `jpeg_compression_level`: integer value in range [1, 100]  representing compression level of submitted images 
(optional, defaults to `95`)
* `persist_predictions`: binary flag to decide if predictions should be collected along with images (required if `enabled`)
* `sampling_strategies`: list of sampling strategies (non-empty list required if `enabled`)
* `batching_strategy`: configuration of labeling batches creation - details below (required if `enabled`)
* `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)

## How to Configure Active Learning

To configure active learning, you need to make a HTTP request to Roboflow with your configuration.

### Set Configuration

To set an active learning configuration, use the following code:

```python
import requests

def set_active_learning_configuration(
    workspace: str,
    project: str,
    api_key: str,
    config: dict,
) -> None:
    response = requests.post(
        f"https://api.roboflow.com/{workspace}/{project}/active_learning?api_key={api_key}",
        json={
            "config": config,
        }
    )
    return response.json()

set_active_learning_configuration(
    workspace="yyy",
    project="zzz",
    api_key="XXX",
    config={
        "enabled": True,
        "persist_predictions": True,
        "batching_strategy": {
            "batches_name_prefix": "my_batches",
            "recreation_interval": "daily",
        },
        "sampling_strategies": [
            {
                "name": "default_strategy",
                "type": "random",
                "traffic_percentage": 0.01, 
            }
        ]
    }
)
```

Where:

1. `workspace` is your workspace name;
2. `project` is your project name;
3. `api_key` is your API key, and;
4. `config` is your active learning configuration.

### Retrieve Existing Configuration

To retrieve an existing active learning configuration, use the following code:

```python
import requests

def get_active_learning_configuration(
    workspace: str,
    project: str,
    api_key: str
) -> None:
    response = requests.get(
        f"https://api.roboflow.com/{workspace}/{project}/active_learning?api_key={api_key}",
    )
    return response.json()
```

Above, replace `workspace` with your workspace name, `project` with your project name, and `api_key` with your API key.

#### Batching strategy

The `batching_strategy` field holds a dictionary with the following configuration options:

* `batches_name_prefix`: A string representing the prefix of batches names created by Active Learning (required)
* `recreation_interval`: One of value `["never", "daily", "weekly", "monthly"]`: representing the interval which is to be used to create separate batches - thanks to that - user can control the flow of labeling batches in time (required).
* `max_batch_images`: Positive integer representing maximum size of batch (applied on top of any strategy limits) to prevent too much data to be collected (optional)

#### Example configuration
```json
{
    "enabled": true,
    "max_image_size": [1200, 1200],
    "jpeg_compression_level": 75,
    "persist_predictions": true,
    "sampling_strategies": [
      {
        "name": "default_strategy",
        "type": "random",
        "traffic_percentage": 0.1
      }
    ],
    "batching_strategy": {
      "batches_name_prefix": "al_batch",
      "recreation_interval": "daily"
    }
}
```

### <a name="detections-number-based-sampling"> Detection number based sampling (for detection)

## Strategy limits
Each strategy can be configured with `limits`: list of values limiting how many images can be collected 
each minute, hour or day. Each entry on that list can hold two values:
* `type`: one of `["minutely", "hourly", "daily"]`: representing the type of limit
* `value`: with limit threshold

Limits are enforced with different granularity, as they are implemented based or either Redis or memory cache (bounded
into single process). Se effectively:
* if Redis cache is used - all instances of `inference` connected to the same Redis service will share limits 
enforcements
* otherwise - memory cache of single instance is used (multiple processes will have their own limits)

Self-hosted `inference` may be connected to your own Redis cache.

## Stubs
One may use `{dataset_name}/0` as `model_id` while making prediction - to use null model for specific project. 
It is going to provide predictions in the following format:
```json
{
    "time": 0.0002442499971948564,
    "is_stub": true,
    "model_id": "asl-poly-instance-seg/0",
    "task_type": "instance-segmentation"
}
```
{% include 'model_id.md' %}
This option, combined with Active Learning (namely `random` sampling strategy), provides a way to start data collection
even prior any model is trained. There are several benefits of such strategy. The most important is building 
the dataset representing the true production distribution, before any model is trained.

Example client usage:
```python
from inference_sdk import InferenceHTTPClient

LOCALHOST_CLIENT = InferenceHTTPClient(
    api_url="http://127.0.0.1:9001",
    api_key="XXX"
)
LOCALHOST_CLIENT.infer(image, model_id="asl-poly-instance-seg/0")
```