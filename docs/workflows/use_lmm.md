# `LMM`

Use Large Language models with `workflows`. With this block in place, one may prompt both `GPT-4V` and `CogVLM` models and combine their outputs with other workflow components, effectively building powerful applications without single line of code written. 

The `LMM` block allows to specify structure of expected output, automatically inject the specification into prompt and parse expected structure into block outputs that are accessible (and can be referred) by other `workflows` components. LMMs may occasionally produce non-parsable results according to specified output structure - in that cases, outputs will be filled with `not_detected` value.

## Step parameters

* `type`: must be `LMM` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `prompt`: must be string of reference to `InferenceParameter` - value holds unconstrained text prompt to LMM model 
(required).   
* `lmm_type`: must be string of reference to `InferenceParameter` - value holds the type of LMM model to be used - 
allowed values: `gpt_4v` and `cog_vlm` (required)
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
* `remote_api_key` - optional string or reference to `InferenceParameter` that holds API key required to
call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v` and
do not require additional API key for CogVLM calls.
* `json_output`: optional `dict[str, str]` (pointing expected output JSON field name to its description)
or reference to `InferenceParameter` with such dict. This field is used to instruct model on expected output 
format. One may not specify field names: `["raw_output", "structured_output", "image", "parent_id"]`, due to the
fact that keys from `json_output` dict will be registered as block outputs (to be referred by other blocks) and
cannot collide with basic outputs of that block. Additional outputs **will only be registered if defined in-place, 
not via `InferenceParameter`).

## Step outputs
* `raw_output` - raw output of LMM for each input image
* `structured_output` - if `json_output` is specified, whole parsed dictionary for each input image will be placed in this field, 
otherwise for each image, empty dict will be returned
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
* for each of `json_output` - dedicated field will be created (with values provided image-major) - and those can be 
referred as normal outputs (`$steps.{step_name}.{field_name}`).

## Important notes
* `CogVLM` can only be used in `self-hosted` API - as Roboflow platform does not support such model. 
Use `inference server start` on a machine with GPU to test that model.