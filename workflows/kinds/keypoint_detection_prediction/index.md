
# `keypoint_detection_prediction` Kind

Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `dict`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `sv.Detections`

## Details


This kind represents single keypoints prediction in form of 
[`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object.

Example:
```
sv.Detections(
    xyxy=array([[        127,         189,         322,         303]]), 
    mask=None, 
    confidence=array([    0.95898]), 
    class_id=array([6]), 
    tracker_id=None, 
    data={
        'class_name': array(['G'], dtype='<U1'),
        'detection_id': array(['51dfa8d5-261c-4dcb-ab30-9aafe9b52379'], dtype='<U36'),
        'parent_id': array(['image.[0]'], dtype='<U9'),
        'image_dimensions': array([[425, 640]]),
        'inference_id': array(['51dfa8d5-261c-4dcb-ab30-9aafe9b52379'], dtype='<U36'),
        'prediction_type': array(['instance-segmentation'], dtype='<U16'),
        'root_parent_id': array(['image.[0]'], dtype='<U9'),
        'root_parent_coordinates': array([[0, 0]]),
        'root_parent_dimensions': array([[425, 640]]),
        'parent_coordinates': array([[0, 0]]),
        'parent_dimensions': array([[425, 640]]),
        'scaling_relative_to_parent': array([1]),
        'scaling_relative_to_root_parent': array([1]),
        'keypoints_class_name': array(),  # variable length array of type object - one 1D array of str for each box
        'keypoints_class_id': array(),  # variable length array of type object - one 1D array of int for each box
        'keypoints_confidence': array(),  # variable length array of type object - one 1D array of float for each box
        'keypoints_xy': array(),  # variable length array of type object - one 2D array for bbox with (x, y) coords
    }
)
```

Prior to [sv.Keypoints(...)](https://supervision.roboflow.com/0.21.0/keypoint/core/) we introduced 
keypoints detection based on [`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object.
The decision was suboptimal so we would need to revert in the future, but for now this is the format of
data for keypoints detection. 

The design of metadata is also suboptimal (as metadata regarding whole image is duplicated across all 
bounding boxes and there is no way on how to save metadata for empty predictions). We
have [GH issue](https://github.com/roboflow/inference/issues/567) to communicate around this
problem.

**Details of additional fields:**

* `detection_id` - unique identifier for each detection, to be used for when dependent elements 
are created based on specific detection (example: Dynamic Crop takes this value as parent id for new image)

* `parent_id` - identifier of image that generated prediction (to be fetched from `WorkflowImageData` object)

* `image_dimensions` - dimensions of image that was basis for prediction - format: `(height, width)`

* `inference_id` - identifier of inference request (optional, relevant for Roboflow models)

* `prediction_type` - type of prediction

* `root_parent_id` - identifier of primary Workflow input that was responsible for downstream prediction 
(to be fetched from `WorkflowImageData` object) - usually identifier of Workflow input placeholder 

* `root_parent_coordinates` - offset regarding origin input - format (`offset_x`, `offset_y`)

* `root_parent_dimensions` - dimensions of origin input image `(height, width)`

* `parent_coordinates` - offset regarding parent - format (`offset_x`, `offset_y`)

* `parent_dimensions` - dimensions of parent image `(height, width)`

* `scaling_relative_to_parent` - scaling factor regarding parent image

* `scaling_relative_to_root_parent` - scaling factor regarding origin input image

* `keypoints_class_name` array of variable size 1D arrays of string with key points class names

* `keypoints_class_id` array of variable size 1D arrays of int with key points class ids

* `keypoints_confidence` array of variable size 1D arrays of float with key points confidence

* `keypoints_xy` array of variable size 2D arrays of coordinates of keypoints in `(x, y)` format

**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output, but **loosing keypoints details** - which can be recovered if output 
JSON field is parsed. Entity details: [KeypointsDetectionInferenceResponse](https://detect.roboflow.com/docs)


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
