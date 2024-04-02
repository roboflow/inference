# `QRCodeDetector`

Detect the location of QR codes in an image.

## Step parameters
* `type`: must be `QRCodeDetector` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)

## Step outputs:
* `predictions` - details of predictions
    * Note: `predictions.data` is a string which is populated with the data contents of the QR code.
* `image` - size of input image, that `predictions` coordinates refers to
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
* `prediction_type` - denoting `qrcode-detection` model

## Format of `predictions`
`predictions` is batch-major list of size `[batch_size, #detections_for_input_image]`.
Each detection is in format:
```json
{
    "parent_id": "uuid_of_parent_element",
    "class": "qr_code",
    "class_id": 0,
    "confidence": 1.0,
    "x": 128.5,
    "y": 327.8,
    "width": 200.0,
    "height": 150.0,
    "detection_id": "uuid_of_detection",
    "data": "here you can find text detected in qr_code"
}
```