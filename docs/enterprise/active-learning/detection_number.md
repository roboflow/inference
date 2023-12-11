Choose specific detections based on count and detection classes. Collect and save images for use in improving your model.

!!! tip
    Review the [Active Learning](../active_learning.md) page for more information about how to use active learning.

This strategy is available for the following model types:

* `object-detection`
* `instance-segmentation`
* `keypoints-detection`

## Configuration

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

## Example

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

Learn how to [configure active learning](../active_learning.md#configuration) for your model.