# Active Learning

Inference lets you collect data automatically from production use cases for use in training new versions of your model. 
This is called active learning.

By collecting data actively that you can later label and use in training, you can gather data representative of the 
environment in which your model is deployed. This information may help you boost model performance.

The core concept:
* The user initiates the creation of an Active Learning configuration within the Roboflow app.
* This configuration is then distributed across all active inference instances, which may include those running against 
video streams and the HTTP API, both on-premises and within the Roboflow platform.
* During runtime, as predictions are generated, images and model predictions (treated as initial annotations) are 
dynamically collected and submitted in batches into user project. These batches are then ready for labeling within the 
Roboflow platform.

To decide which datapoints should be collected, Active Learning components use sampling strategies that can be 
defined by users.

## Active Learning configuration
Active learning configuration contains **sampling strategies** and other fields influencing the feature behaviour.

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

#### Batching strategy
`batching_strategy` field holds a dictionary with the following configuration options:
* `batches_name_prefix`: string representing the prefix of batches names created by Active Learning (required)
* `recreation_interval`: one of value `["never", "daily", "weekly", "monthly"]`: representing the interval which
is to be used to create separate batches - thanks to that - user can control the flow of labeling batches in time
(required)
* `max_batch_images`: positive integer representing maximum size of batch (applied on top of any strategy limits)
to prevent too much data to be collected (optional)

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

## Sampling strategies

Every user-defined strategy must possess a unique name within a given configuration. In addition to the name, 
each strategy is distinguished by its specified type. Various types of strategies are available for utilization.

List of supported strategies:
* [random sampling](#random-sampling)
* [close-to-threshold sampling](#close-to-threshold-sampling)
* [classes based sampling](#classes-based-sampling)
* [detections number based sampling](#detections-number-based-sampling)

###  <a name="random-sampling"></a> Random sampling strategy
This strategy should be used to randomly select data to be saved for future labeling. 

#### Applicable model types
* `stub`
* `classification`
* `object-detection`
* `instance-segmentation`
* `keypoints-detection`

#### Configuration options
* `name`: user-defined name of the strategy - must be non-empty and unique within all strategies defined in a 
single configuration (required)
* `type`: with value `random` is used to identify random sampling strategy (required)
* `traffic_percentage`: float value in range [0.0, 1.0] defining the percentage of traffic to be persisted (required)
* `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)
* `limits`: definition of limits for data collection within a specific strategy

#### Configuration example
```json
{
    "name": "my_random_sampling",
    "type": "random",
    "traffic_percentage": 0.01,
    "tags": ["my_tag_1", "my_tag_2"],
    "limits": [
        {"type": "minutely", "value": 10},
        {"type": "hourly", "value": 100},
        {"type": "daily", "value": 1000}
    ]
}
```

### <a name="close-to-threshold-sampling"></a> Close to threshold sampling
Sampling method intended for selecting data points that lead to specific prediction confidences for particular classes. 
This method is applicable to both detection and classification models, although the behavior may vary slightly 
between the two.

#### Applicable model types
* `classification`
* `object-detection`
* `instance-segmentation`
* `keypoints-detection`

#### Configuration options
* `name`: user-defined name of the strategy - must be non-empty and unique within all strategies defined in a 
single configuration (required)
* `type`: with value `close_to_threshold` is used to identify close to threshold sampling strategy (required)
* `selected_class_names`: list of class names to consider during sampling; if not provided, all classes can be sampled. (Optional)
* `threshold` and `epsilon`: Represent the center and radius for the confidence range that triggers sampling. Both are
to be float values in range [0.0, 1.0]. For example, if one aims to obtain datapoints where the classifier is highly 
confident (0.8, 1.0), set threshold=0.9 and epsilon=0.1. Note that this is limited to outcomes from model 
post-processing and threshold filtering - hence not all model predictions may be visible at the level of Active Learning 
logic. (required)
* `probability`: Fraction of datapoints matching sampling criteria that will be persisted. It is meant to be float 
value in range [0.0, 1.0] (required)
* `minimum_objects_close_to_threshold`: (used for detection predictions only) Specify how many detected objects from 
selected classes must be close to the threshold to accept the datapoint. If given - must be integer value >= 1. 
(Optional - with default to `1`)
* `only_top_classes`: (used for classification predictions only) Flag to decide whether only the `top` or 
`predicted_classes` (for multi-class/multi-label cases, respectively) should be considered. This helps avoid sampling 
based on non-leading classes in predictions. Default: `True`.
* `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)

#### Configuration example
```json
{
    "name": "hard_examples",
    "type": "close_to_threshold",
    "selected_class_names": ["a", "b"],
    "threshold": 0.25,
    "epsilon": 0.1,
    "probability": 0.5,
    "tags": ["my_tag_1", "my_tag_2"],
    "limits": [
        {"type": "minutely", "value": 10},
        {"type": "hourly", "value": 100},
        {"type": "daily", "value": 1000}
    ]
}
```

### <a name="classes-based-sampling"> Classes based sampling (for classification)
Sampling method employed to selectively choose specific classes from classifier predictions.

#### Applicable model types
* `classification`

#### Configuration options
* `name`: user-defined name of the strategy - must be non-empty and unique within all strategies defined in a 
single configuration (required)
* `type`: with value `classes_based` is used to identify close to threshold sampling strategy (required)
* `selected_class_names`: list of class names to consider during sampling - (required)
* `probability`: fraction of datapoints that matches sampling criteria that will be persisted. It is meant to be float 
value in range [0.0, 1.0] (required)
* `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)

#### Configuration example
```json
 {
    "name": "underrepresented_classes",
    "type": "classes_based",
    "selected_class_names": ["cat"],
    "probability": 1.0,
    "tags": ["hard-classes"],
    "limits": [
        {"type": "minutely", "value": 10},
        {"type": "hourly", "value": 100},
        {"type": "daily", "value": 1000}
    ]
}
```

### <a name="detections-number-based-sampling"> Detection number based sampling (for detection)
Sampling method employed for selectively choosing specific detections based on count and detection classes.

#### Applicable model types
* `object-detection`
* `instance-segmentation`
* `keypoints-detection`

#### Configuration options
* `name`: user-defined name of the strategy - must be non-empty and unique within all strategies defined in a 
single configuration (required)
* `type`: with value `detections_number_based` is used to identify close to threshold sampling strategy (required)
* `selected_class_names`: list of class names to consider during sampling; if not provided, all classes can be sampled. (Optional)
* `probability`: fraction of datapoints that matches sampling criteria that will be persisted. It is meant to be float 
value in range [0.0, 1.0] (required)
* `more_than`: minimal number of detected objects - if given it is meant to be integer >= 0 
(optional - if not given - lower limit is not applied)
* `less_than`: maximum number of detected objects - if given it is meant to be integer >= 0 
(optional - if not given - upper limit is not applied)
* **NOTE:** if both `more_than` and `less_than` is not given - any number of matching detections will match the 
sampling condition
* `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)

#### Configuration example
```json
{
    "name": "multiple_detections",
    "type": "detections_number_based",
    "probability": 0.2,
    "more_than": 3,
    "tags": ["crowded"],
    "limits": [
        {"type": "minutely", "value": 10},
        {"type": "hourly", "value": 100},
        {"type": "daily", "value": 1000}
    ]
}
```

## Active Learning sampling
The sampling procedure contains few steps:
* at first, sampling methods are evaluated to decide which ones are applicable to the image and prediction (evaluation
happens in the order of definition)
* global limit for batch (defined in `batching_strategy`) is checked - its violation terminates Active Learning attempt
* matching methods are checked against limits defined within their configurations - first method with matching limit
is selected

Once datapoint is selected and there is no limit violation - it will be saved into Roboflow platform with tags 
relevant for specific strategy (and global tags defined at the level of Active Learning configuration).

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

## How to configure Active Learning using Roboflow API directly?

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

# Example usage
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
