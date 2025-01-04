# Active Learning

Active Learning is a process of iterative improvement of model by retraining models on dataset that grows over time.
This process includes data collection (usually with smart selection of datapoints that model would most benefit from),
labeling, model re-training, evaluation and deployment - to close the circle and start new iteration.

Elements of that process can be partially or fully automated - providing an elegant way of improving dataset over time, 
which is important to ensure good quality of model predictions over time (as the data distribution may change and a model
trained on old data may not be performant facing the new one). At Roboflow, we've brought automated data collection 
mechanism - which is the foundational building block for Active Learning  -- to the platform.

## Where to start?

We suggest clients apply the following strategy to train their models. If it's applicable - start from a small, good 
quality dataset labeled manually (making sure that the test set is representative of the problem to be solved) and train 
an initial model. Once that is done - deploy your model, enabling Active Learning data collection, and gradually increase 
the size of your dataset with data collected in production environment.

Alternatively, it is also possible to start the project with [a Universe model](https://universe.roboflow.com/). Then,
for each request you can specify `active_learning_target_dataset` - pointing to the project where the data should be 
saved. This way, if you find a model that meets your minimal quality criteria, you may start generating valuable 
predictions from day zero, while collecting good quality data to train even better models in the future.

## How Active Learning data collection works? 

Here is the standard workflow for Active Learning data collection:

* The user initiates the creation of an Active Learning configuration within the Roboflow app.
* This configuration is then distributed across all active inference instances, which may include those running against video streams and the HTTP API, both on-premises and within the Roboflow platform.
* During runtime, as predictions are generated, images and model predictions (treated as initial annotations) are dynamically collected and submitted in batches into user project. These batches are then ready for labeling within the Roboflow platform.

How active learning works with Inference is configured in your server active learning configuration. [Learn how to configure active learning](#active-learning-configuration).

Active learning can be disabled by setting `ACTIVE_LEARNING_ENABLED=false` in the environment where you run `inference`.

## Usage patterns
Active Learning data collection may be combined with different components of the Roboflow ecosystem. In particular:

- the `inference` Python package can be used to get predictions from the model and register them on the Roboflow platform
  - one may want to use `InferencePipeline` to get predictions from video and register its video frames using Active Learning
- self-hosted `inference` server - where data is collected while processing requests
- Roboflow hosted `inference` - where you let us make sure you get your predictions and data registered. No 
infrastructure needs to run on your end, we take care of everything
  - [Roboflow `workflows`](../../workflows/about.md) - our newest feature - supports [`Roboflow Dataset Upload block`](/workflows/blocks/roboflow_dataset_upload.md)


## Sampling Strategies

`inference` makes it possible to configure the way data is selected for registration. One may configure one or more sampling
strategies during Active Learning configuration. We support five strategies for sampling image data for use in training new model versions. 
These strategies are:

* [Random sampling](./random_sampling.md): Images are collected at random.
* [Close-to-threshold](./close_to_threshold_sampling.md): Collect data close to a given threshold.
* [Detection count-based](./detection_number.md) (Detection models only): Collect data with a specific number of detections returned by a detection model.
* [Class-based](./classes_based.md) (Classification models only): Collect data with a specific class returned by a classification model.

## How Data is Sampled

When you run Inference with an active learning configuration, the following steps are run:

1. Sampling methods are evaluated to decide which ones are applicable to the image and prediction (evaluation happens in the order of definition in your configuration).
2. A global limit for batch (defined in `batching_strategy`) is checked. Its violation terminates Active Learning attempt.
* Matching methods are checked against limits defined within their configurations. The first method with matching limit is selected.

Once a datapoint is selected and there is no limit violation, it will be saved into Roboflow platform with tags relevant for specific strategy (and global tags defined at the level of Active Learning configuration).

## Active Learning Configuration

One may choose to configure their Active Learning with the Roboflow app UI by navigating to the `Active Learning` panel.
Alternatively, requests to Roboflow API may be sent with custom configuration. Here is how to configure Active Learning
directly through the API.

### Configuration options

* `enabled`: boolean flag to enable / disable the configuration (required) - `{"enabled": false}` is minimal valid config
* `max_image_size`: two element list with positive integers (height, width) enforcing down-sizing (with aspect-ratio
preservation) of images before submission into Roboflow platform (optional)
* `jpeg_compression_level`: integer value in range [1, 100]  representing compression level of submitted images 
(optional, defaults to `95`)
* `persist_predictions`: binary flag to decide if predictions should be collected along with images (required if `enabled`)
* `sampling_strategies`: list of sampling strategies (non-empty list required if `enabled`)
* `batching_strategy`: configuration of labeling batches creation - details below (required if `enabled`)
* `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)

### Batching strategy

The `batching_strategy` field holds a dictionary with the following configuration options:

* `batches_name_prefix`: A string representing the prefix of batch names created by Active Learning (required)
* `recreation_interval`: One of `["never", "daily", "weekly", "monthly"]`: representing the interval which is to be used to create separate batches. This parameter allows the user to control the flow of labeling batches over time (required).
* `max_batch_images`: Positive integer representing the maximum size of the batch (applied on top of any strategy limits) to prevent too much data from being collected (optional)


### Strategy limits
Each strategy can be configured with `limits`: list of values limiting how many images can be collected 
each minute, hour or day. Each entry on that list can hold two values:
* `type`: one of `["minutely", "hourly", "daily"]`: representing the type of limit
* `value`: with limit threshold

Limits are enforced with different granularity, as they are implemented based or either Redis or memory cache (bounded
into a single process). So, effectively:
* if the Redis cache is used - all instances of `inference` connected to the same Redis service will share limit 
enforcements
* otherwise, the memory cache of a single instance is used (multiple processes will have their own limits)

Self-hosted `inference` may be connected to your own Redis cache.

### Example configuration
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
        "traffic_percentage": 0.1,
        "limits": [{"type": "daily", "value": 100}]
      }
    ],
    "batching_strategy": {
      "batches_name_prefix": "al_batch",
      "recreation_interval": "daily"
    }
}
```

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
                "limits": [{"type": "daily", "value": 100}]
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
import cv2
from inference_sdk import InferenceHTTPClient

image = cv2.imread("<path_to_your_image>")
LOCALHOST_CLIENT = InferenceHTTPClient(
    api_url="http://127.0.0.1:9001",
    api_key="XXX"
)
LOCALHOST_CLIENT.infer(image, model_id="asl-poly-instance-seg/0")
```

## Parameters of requests to `inference` server influencing the Active Learning data collection
There are a few parameters that can be added to request to influence how data collection works, in particular:

- `disable_active_learning` - to disable functionality at the level of a single request (if for some reason you do not 
want input data to be collected - useful for testing purposes)
- `active_learning_target_dataset` - making inference from a specific model (let's say `project_a/1`), when we want
to save data in another project `project_b` - the latter should be pointed to by this parameter. **Please remember that
you cannot use incompatible types of models in `project_a` and `project_b`; if that is the case, data will not be 
registered. For instance, classification predictions cannot be registered in detection-based projects.** You are free to mix
 tasks like object-detection, instance-segmentation, or keypoints detection, but naturally not every detail of
the required label may be available from prediction.


Visit [Inference SDK docs](../../inference_helpers/inference_sdk.md) to learn more.
