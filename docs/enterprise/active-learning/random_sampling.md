Randomly select data to be saved for future labeling.

!!! tip
    Review the [Active Learning](../active_learning.md) page for more information about how to use active learning.

This strategy is available for the following model types:

* `stub`
* `classification`
* `object-detection`
* `instance-segmentation`
* `keypoints-detection`

## Configuration

* `name`: user-defined name of the strategy - must be non-empty and unique within all strategies defined in a 
single configuration (required)
* `type`: with value `random` is used to identify random sampling strategy (required)
* `traffic_percentage`: float value in range [0.0, 1.0] defining the percentage of traffic to be persisted (required)
* `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)
* `limits`: definition of limits for data collection within a specific strategy

## Example

Here is an example of a configuration manifest for random sampling strategy:

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

Learn how to [configure active learning](../active_learning.md#configuration) for your model.