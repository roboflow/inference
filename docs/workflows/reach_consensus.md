# `DetectionsConsensus`

Combine predictions from multiple detections models to make a decision about object presence.

Steps checks for object presence (according to configurable criteria), combines detections and
decides on requested objects presence (based on overlap of predictions from different models). It works for 
`object-detection`, `instance-segmentation` and `keypoint-detection` models, but consensus output is only
applied at detections level. 

Step executes following operations (in order):
* get only the predictions from `classes_to_consider` (if specified)
* for every prediction finds predictions with max overlap from all other sources (at most one per source) that reaches 
`iou_threshold` 
* for each group of overlapping predictions from different sources - if the size of group is at least
`required_votes` and merged boxe meet `confidence` threshold - those are discarded from the pool of detections to be 
picked up and are merged into element of `predictions` output that can be called consensus output. 
`class_aware` parameter decides if class names matter while merging - should be `False` when different class names are 
produced by different models  but the visual concept that models predict is the same.
* merge is done based on `detections_merge_confidence_aggregation` and `detections_merge_coordinates_aggregation` 
parameters that control how to pick the merged box class, confidence and box coordinates
* once all elements of consensus outputs are ready, the step prepares `object_present` status
and `presence_confidence` that form a summary of consensus output. One may state `required_objects`
as integer or dict mapping class name to required instance of objects. In the final state, the step
logic will check if required number of objects (possibly from different classes) are detected in consensus
output. If that's the case - `object_present` field will be `True` and `presence_confidence` will be calculated
(using `presence_confidence_aggregation` method). Otherwise - `presence_confidence` will be an empty dict.
In the case of `class_aware=False`:
  * when `required_objects` is dict with class to count mapping - effective `required_objects` will be sum of dictionary 
  values
  * the `presence_confidence` will hold `any_object` key with confidence aggregated among all merged detections. 

![How consensus block works?](./assets/detection_consensus_step_diagram.jpg)

## Step parameters
* `type`: must be `DetectionsConsensus` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `predictions`: list of selectors pointing to outputs of detections models output of the detections model: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`, `YoloWorld`] (required, must contain at least 2 elements)
* `required_votes`: number of models that must agree on the detection - integer or selector pointing at
`InferenceParameter` (required)
* `class_aware`: flag deciding if class names are taken into account when finding overlapping bounding boxes
from multiple models and object presence check. Can be `bool` or selector to `InferenceParameter`. Default: `True`
* `iou_threshold`: optional float value in range [0, 1] with IoU threshold that must be meet to consider
two bounding boxes overlapping. Can be float or selector to `InferenceParameter`. Default: `0.3`.
* `confidence`: optional float value in range [0, 1] minimal confidence of **aggregated** detection that must be met to 
be taken into account in presence assessment and consensus procedure. For prior-consensus filtering - use
confidence threshold at model level or `DetectionsFilter`. Default: `0.0`.
* `classes_to_consider`: Optional list of classes to consider in consensus procedure.
Can be list of `str` or selector to `InferenceParameter`. Default: `None` - in this case 
classes filtering of predictions will not be enabled.
* `required_objects` - If given, it holds the number of objects that must be present in merged results, to assume that 
object presence is reached. Can be selector to `InferenceParameter`, integer value or dictionary with mapping of class name into
minimal number of merged detections of given class to assume consensus.
* `presence_confidence_aggregation` - mode dictating aggregation of confidence scores
and classes both in case of object presence deduction procedure. One of `average`, `max`, `min`. Default: `max`.
* `detections_merge_confidence_aggregation` - mode dictating aggregation of confidence scores
and classes both in case of boxes consensus procedure. 
One of `average`, `max`, `min`. Default: `average`. While using for merging overlapping boxes, 
against classes - `average` equals to majority vote, `max` - for the class of detection with max confidence,
`min` - for the class of detection with min confidence.
* `detections_merge_coordinates_aggregation` - mode dictating aggregation of bounding boxes. One of `average`, `max`, `min`. 
Default: `average`. `average` means taking mean from all boxes coordinates, `min` - taking smallest box, `max` - taking 
largest box.

## Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines (can be `undefined` if all sources of predictions give no prediction)
* `object_present` - for each input image, boolean flag with information whether or not
objects specified in config are present
* `presence_confidence` - for each input image, for each present class - aggregated confidence indicating presence
of objects
* `prediction_type` - denoting `object-detection` prediction (as this format is effective even if other detections 
models are combined)