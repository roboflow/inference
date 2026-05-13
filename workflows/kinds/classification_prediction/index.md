
# `classification_prediction` Kind

Predictions from classifier

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `dict`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `dict`

## Details


This kind represent predictions from Classification Models.

Examples:
```
# in case of multi-class classification
{
    "image": {"height": 128, "width": 256},
    "predictions": [{"class_name": "A", "class_id": 0, "confidence": 0.3}],
    "top": "A",
    "confidence": 0.3,
    "parent_id": "some",
    "prediction_type": "classification",
    "inference_id": "some",
    "root_parent_id": "some",
}

# in case of multi-label classification
{
    "image": {"height": 128, "width": 256},
    "predictions": {
        "a": {"confidence": 0.3, "class_id": 0},
        "b": {"confidence": 0.3, "class_id": 1},
    }
    "predicted_classes": ["a", "b"],
    "parent_id": "some",
    "prediction_type": "classification",
    "inference_id": "some",
    "root_parent_id": "some",
}
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
