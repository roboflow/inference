# `BarcodeDetection`

Detect the location and value barcodes in an image.

## Step parameters
* `type`: must be `BarcodeDetection` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)

## Step outputs:
* `predictions` - details of predictions
    * Note: `predictions.data` is a string which is populated with the data contents of the barcode.
* `image` - size of input image, that `predictions` coordinates refers to
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
* `prediction_type` - denoting `barcode-detection` model