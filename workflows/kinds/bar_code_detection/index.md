
# `bar_code_detection` Kind

Prediction with barcode detection

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `dict`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `sv.Detections`

## Details


This kind represents batch of predictions regarding barcodes location and data their provide.

Example:
```
sv.Detections(
    xyxy=array([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    mask=None, 
    confidence=array([    1.0, 1.0, 1.0, 1.0]), 
    class_id=array([2, 7, 2, 0]), 
    tracker_id=None, 
    data={
        'class_name': array(['barcode', 'barcode', 'barcode', 'barcode'], dtype='<U13')
        'detection_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'image_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'inference_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'prediction_type': array([
            'barcode-detection', 'barcode-detection', 
            'barcode-detection', 'barcode-detection'
        ], dtype='<U16'),
        'root_parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'root_parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'root_parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'scaling_relative_to_parent': array([1, 1, 1, 1]),
        'scaling_relative_to_root_parent': array([1, 1, 1, 1]),
        'data': np.array(['qr-code-1-data', 'qr-code-2-data', 'qr-code-3-data', 'qr-code-4-data'])
    }
)
```

As you can see, we have extended the standard set of metadata for predictions maintained by `supervision`.
Adding this metadata is needed to ensure compatibility with blocks from `roboflow_core` plugin.

The design of metadata is suboptimal (as metadata regarding whole image is duplicated across all 
bounding boxes and there is no way on how to save metadata for empty predictions). We
have [GH issue](https://github.com/roboflow/inference/issues/567) to communicate around this
problem.

**Details of additional fields:**

* `detection_id` - unique identifier for each detection, to be used for when dependent elements 
are created based on specific detection (example: Dynamic Crop takes this value as parent id for new image)

* `parent_id` - identifier of image that generated prediction (to be fetched from `WorkflowImageData` object)

* `image_dimensions` - dimensions of image that was basis for prediction - format: `(height, width)`

* `inference_id` - identifier of inference request (optional, relevant for Roboflow models)

* `prediction_type` - type of prediction

* `root_parent_id` - identifier of primary Workflow input that was responsible for downstream prediction 
(to be fetched from `WorkflowImageData` object) - usually identifier of Workflow input placeholder 

* `root_parent_coordinates` - offset regarding origin input - format (`offset_x`, `offset_y`)

* `root_parent_dimensions` - dimensions of origin input image `(height, width)`

* `parent_coordinates` - offset regarding parent - format (`offset_x`, `offset_y`)

* `parent_dimensions` - dimensions of parent image `(height, width)`

* `scaling_relative_to_parent` - scaling factor regarding parent image

* `scaling_relative_to_root_parent` - scaling factor regarding origin input image

* `data` - extracted barcode

**SERIALISATION:**
Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [ObjectDetectionInferenceResponse](https://detect.roboflow.com/docs)


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
