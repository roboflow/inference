# `ClipComparison`

Compare CLIP image and text embeddings.

## Step parameters
* `type`: must be `ClipComparison` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `text`: reference to `InputParameter` of list of texts to compare against `image` using Clip model

## Step outputs:
* `similarity` - for each element of `image` - list of float values representing similarity to each element of `text`
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines