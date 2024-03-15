# `LMMForClassification`

Use LMMs (both `GPT-4V` and `CogVLM` models) as zero-shot classification blocks.

## Step parameters
* `type`: must be `LMM` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `lmm_type`: must be string of reference to `InferenceParameter` - value holds the type of LMM model to be used - 
allowed values: `gpt_4v` and `cog_vlm` (required)
* `classes` - non-empty list of class names (strings) or reference to `InferenceParameter` that holds this value. 
Classes are presented to LMM in prompt and model is asked to produce structured classification output (required).
* `remote_api_key` - optional string or reference to `InferenceParameter` that holds API key required to
call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v` and
do not require additional API key for CogVLM calls.
* `lmm_config`: (optional) structure that has the following schema:
```json
{
  "max_tokens": 450,
  "gpt_image_detail": "low",
  "gpt_model_version": "gpt-4-vision-preview"
}
```
to control inner details of LMM prompting. All parameters now are suited to control GPT API calls. Default for
max tokens is `450`, `gpt_image_detail` default is `auto` (allowed values: `low`, `auto`, `high`), 
`gpt_model_version` is `gpt-4-vision-preview`.

## Step outputs
* `raw_output` - raw output of LMM for each input image
* `top` - name of predicted class for each image
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
* `prediction_type` - type of prediction output: `classification`

## Important notes
* `CogVLM` can only be used in `self-hosted` API - as Roboflow platform does not support such model. 
Use `inference server start` on a machine with GPU to test that model.