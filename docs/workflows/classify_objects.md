# `ClassificationModel`

Run a classification model.

## Step parameters
* `type`: must be `ClassificationModel` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `model_id`: must be either valid Roboflow model ID or selector to  input parameter (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `disable_active_learning`: optional boolean flag to control Active Learning at request level - can be selector to 
input parameter 
* `confidence`: optional float value in range [0, 1] with threshold - can be selector to 
input parameter 
* `active_learning_target_dataset`: optional name of target dataset (or reference to `InferenceParemeter`) 
dictating that AL should collect data to a different dataset than the one declared by the model

## Step outputs:
* `predictions` - details of predictions
* `top` - top class
* `confidence` - confidence of prediction
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
* `prediction_type` - denoting `classification` model


## Format of `predictions`
`predictions` is batch-major list of size `[batch_size, #classes_in_the_model]`.
Each prediction for single class is in format:
```json
{
    "class": "class_determined_by_model",
    "class_id": 0,
    "confidence": 1.0,
}
```