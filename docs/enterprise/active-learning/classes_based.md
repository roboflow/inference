Collect and save images that match a class from a classifier prediction for use in improving your model.

!!! tip

    Review the [Active Learning](../active_learning) page for more information about how to use active learning.

This strategy is available for the following model types:

- `classification`

## Configuration

- `name`: user-defined name of the strategy - must be non-empty and unique within all strategies defined in a
  single configuration (required)
- `type`: with value `classes_based` is used to identify close to threshold sampling strategy (required)
- `selected_class_names`: list of class names to consider during sampling - (required)
- `probability`: fraction of datapoints that matches sampling criteria that will be persisted. It is meant to be float
  value in range [0.0, 1.0] (required)
- `tags`: list of tags (each contains 1-64 characters from range `a-z, A-Z, 0-9, and -_:/.[]<>{}@`) (optional)

## Example

Here is an example of a configuration manifest for the close to threshold sampling strategy:

```json
{
  "name": "underrepresented_classes",
  "type": "classes_based",
  "selected_class_names": ["cat"],
  "probability": 1.0,
  "tags": ["hard-classes"],
  "limits": [
    { "type": "minutely", "value": 10 },
    { "type": "hourly", "value": 100 },
    { "type": "daily", "value": 1000 }
  ]
}
```

Learn how to [configure active learning](../active_learning.md#configuration) for your model.
