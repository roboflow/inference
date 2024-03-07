# `AbsoluteStaticCrop` and `RelativeStaticCrop`

Crop regions of interest in an image.

You can use absolute coordinates (integer pixel values) or relative coordinates (fraction of width and height in range [0.0, 1.0]).

## Step parameters
* `type`: must be `AbsoluteStaticCrop` / `RelativeStaticCrop` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `x_center`: OX center coordinate of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`
* `y_center`: OY center coordinate of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`
* `width`: width of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`
* `height`: height of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`

## Step outputs:
* `crops` - `image` cropped based on step parameters
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines