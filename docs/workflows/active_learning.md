# `ActiveLearningDataCollector`

Collect data and predictions that flow through workflows for use in active learning.

This block is built on the foundations of Roboflow Active Learning capabilities implemented in 
[`active_learning` module](../../core/active_learning/README.md).

## Step parameters
* `type`: must be `ActiveLearningDataCollector` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `predictions` - selector pointing to outputs of detections models output of the detections model: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`, `YoloWorld`] (then use `$steps.<det_step_name>.predictions`)
or outputs of classification [`ClassificationModel`] (then use `$steps.<cls_step_name>.top`) (required)
* `target_dataset` - name of Roboflow dataset / project to be used as target for collected data (required)
* `target_dataset_api_key` - optional API key to be used for data registration. This may help in a scenario when data
are to be registered cross-workspaces. If not provided - the API key from a request would be used to register data (
applicable for Universe models predictions to be saved in private workspaces and for models that were trained in the same 
workspace (not necessarily within the same project)).
* `disable_active_learning` - boolean flag that can be also reference to input - to arbitrarily disable data collection
for specific request - overrides all AL config. (optional, default: `False`)
* `active_learning_configuration` - optional configuration of Active Learning data sampling in the exact format provided
in [`active_learning` docs](../../core/active_learning/README.md)

## Step outputs
No outputs are declared. This sep is supposed to cause side effect in form of data sampling and registration. 

## Important Notes

* This block is implemented in non-async way - which means that in certain cases it can block event loop causing
parallelization not feasible. This is not the case when running in `inference` HTTP container. At Roboflow 
hosted platform - registration cannot be executed as background task - so its duration must be added into expected 
latency
* Be careful in enabling / disabling AL at the level of steps - remember that when 
predicting from each model, `inference` HTTP API tries to get Active Learning config from the project that model
belongs to and register datapoint. To prevent that from happening - model steps can be provided with 
`disable_active_learning=True` parameter. Then the only place where AL registration happens is `ActiveLearningDataCollector`.
* Be careful with names of sampling strategies if you define Active Learning configuration - 
you should keep them unique not only within a single config, but globally in project - otherwise limits accounting may
not work well.