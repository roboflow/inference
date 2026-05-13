
# `detection` Kind

Single element of detections-based prediction (like `object_detection_prediction`)

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `Tuple[list, Optional[list], Optional[float], Optional[float], Optional[int], dict]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[float], Optional[int], dict]`

## Details


This kind represents single detection in prediction from a model that detects multiple elements
(like object detection or instance segmentation model). It is represented as a tuple
that is created from `sv.Detections(...)` object while iterating over its content. `workflows`
utilises `data` property of `sv.Detections(...)` to keep additional metadata which will be available
in the tuple. Some properties may not always be present. Take a look at documentation of 
`object_detection_prediction`, `instance_segmentation_prediction`, `keypoint_detection_prediction`
kinds to discover which additional metadata are available.

More technical details about 
[iterating over `sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections)


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
