Select data points that lead to specific prediction confidences for particular classes. 

This method is applicable to both detection and classification models, although the behavior may vary slightly between the two.

!!! tip
    Review the [Active Learning](../active_learning.md) page for more information about how to use active learning.

This strategy is available for the following model types:

* `classification`
* `object-detection`
* `instance-segmentation`
* `keypoints-detection`

## Configuration

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

## Example

Here is an example of a configuration manifest for the close to threshold sampling strategy:

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

Learn how to [configure active learning](../active_learning.md#configuration) for your model.