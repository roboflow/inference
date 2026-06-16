from inference.core.workflows.execution_engine.entities.types import Kind

TENSOR_NATIVE_EMBEDDING_KIND_DOCS = """
This kind represents a vector embedding. It is a list of floating point numbers.

Embeddings are used in various machine learning tasks like clustering, classification,
and similarity search. They are used to represent data in a continuous, low-dimensional space.

Typically, vectors that are close to each other in the embedding space are considered similar.
"""
TENSOR_NATIVE_EMBEDDING_KIND = Kind(
    name="embedding",
    description="A list of floating point numbers representing a vector embedding.",
    docs=TENSOR_NATIVE_EMBEDDING_KIND_DOCS,
    serialised_data_type="List[float]",
    internal_data_type="torch.Tensor",
)


TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND_DOCS = """
This kind represent predictions from Classification Models. Internal representation is 
based on `inference_models.ClassificationPrediction` and `inference_models.MultiLabelClassificationPrediction`
respectivelly. Below one can find external - serialised representation.

Examples:
```
# in case of multi-class classification
{
    "image": {"height": 128, "width": 256},
    "predictions": [{"class_name": "A", "class_id": 0, "confidence": 0.3}],
    "top": "A",
    "confidence": 0.3,
    "parent_id": "some",
    "prediction_type": "classification",
    "inference_id": "some",
    "root_parent_id": "some",
}

# in case of multi-label classification
{
    "image": {"height": 128, "width": 256},
    "predictions": {
        "a": {"confidence": 0.3, "class_id": 0},
        "b": {"confidence": 0.3, "class_id": 1},
    }
    "predicted_classes": ["a", "b"],
    "parent_id": "some",
    "prediction_type": "classification",
    "inference_id": "some",
    "root_parent_id": "some",
}
```
"""
TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND = Kind(
    name="classification_prediction",
    description="Predictions from classifier",
    docs=TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="Union[inference_models.ClassificationPrediction, inference_models.MultiLabelClassificationPrediction]",
)

TENSOR_NATIVE_DETECTION_KIND_DOCS = """
This kind represents single detection in prediction from a model that detects multiple elements
(like object detection or instance segmentation model). It is represented as a tuple
that is created from `inference_models.Detections(...)` object while iterating over its content. `workflows`
utilises `bboxes_metadata` as well as `image_metadata` property of `inference_models.Detections(...)` to keep
additional metadata which will be available in the tuple. Some properties may not always be present. Take a look at
documentation of `object_detection_prediction`, `instance_segmentation_prediction`, `keypoint_detection_prediction`
kinds to discover which additional metadata are available.

**Per-box metadata keys** (read from `bboxes_metadata[i]`, the entry for the i-th box):

* `class` - per-box class label string. When present it overrides the class name resolved from
`image_metadata[CLASS_NAMES_KEY]` keyed by `class_id` (so producers may carry an arbitrary
per-box label, e.g. VLM-as-detector labels or recognised OCR text). When absent, the class name
is resolved from `image_metadata[CLASS_NAMES_KEY][class_id]`.

* `tracker_id` - per-box tracker identifier (present once a tracking block has run).

* `detection_id` - per-box unique identifier (see `object_detection_prediction` for details).

* `text` - per-box recognised text (used by OCR producers); note that the serialised
representation surfaces recognised text under the `class` key, not `text`.
"""

TENSOR_NATIVE_DETECTION_KIND = Kind(
    name="detection",
    description="Single element of detections-based prediction (like `object_detection_prediction`)",
    docs=TENSOR_NATIVE_DETECTION_KIND_DOCS,
    serialised_data_type="Tuple[list, Optional[list], Optional[float], Optional[float], Optional[int], dict, dict]",
    internal_data_type="Tuple[torch.Tensor, Optional[torch.Tensor], Optional[float], Optional[float], Optional[int], dict, dict]",
)


TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND_DOCS = """
This kind represents single object detection prediction in form of `inference_models.Detections` object.

Example:
```
inference_models.Detections(
    xyxy=torch.Tensor([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    confidence=torch.Tensor([    0.84955,     0.74344,     0.45636,     0.86537]), 
    class_id=torch.Tensor([2, 7, 2, 0]), 
    tracker_id=None, 
    image_metadata={
        "class_names": {0: "car", 1: "truck"},
        "image_dimensions": [425, 640],
        "parent_id": "image.[0]",
        "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
        "prediction_type": "object-detection",
        "root_parent_id": "image.[0]",
        "root_parent_coordinates": [0, 0],
        "root_parent_dimensions": [425, 640],
        "parent_coordinates": [0, 0],
        "parent_dimensions": [425, 640],
        "scaling_relative_to_parent": 1.0,
        "scaling_relative_to_root_parent": 1.0,
    }
    bboxes_metadata={
        'detection_id': [
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ],
    }
)
```   
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


**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [ObjectDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND = Kind(
    name="object_detection_prediction",
    description="Prediction with detected bounding boxes in form of inference_models.Detections(...) object",
    docs=TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="inference_models.Detections",
)


TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS = """
This kind represents single object detection prediction in form of `inference_models.InstanceDetections` object.

Example:
```
inference_models.InstanceDetections(
    xyxy=torch.Tensor([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    mask=InstancesRLEMasks(...),
    confidence=torch.Tensor([    0.84955,     0.74344,     0.45636,     0.86537]),
    class_id=torch.Tensor([2, 7, 2, 0]),
    tracker_id=None,
    image_metadata={
        "class_names": {0: "car", 1: "truck"},
        "image_dimensions": [425, 640],
        "parent_id": "image.[0]",
        "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
        "prediction_type": "instance-segmentation",
        "root_parent_id": "image.[0]",
        "root_parent_coordinates": [0, 0],
        "root_parent_dimensions": [425, 640],
        "parent_coordinates": [0, 0],
        "parent_dimensions": [425, 640],
        "scaling_relative_to_parent": 1.0,
        "scaling_relative_to_root_parent": 1.0,
    }
    bboxes_metadata={
        'detection_id': [
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ],
    }
)
```
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

**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output (up to RLE representation). Entity details: [InstanceSegmentationInferenceResponse](https://detect.roboflow.com/docs)
"""
TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="instance_segmentation_prediction",
    description="Prediction with detected bounding boxes and segmentation masks in form of inference_models.InstanceDetections(...) object",
    docs=TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="inference_models.InstanceDetections",
)


TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS = """
This kind represents single instance segmentation prediction in form of an
`inference_models.InstanceDetections` object whose masks are RLE-encoded
(`mask` is an `inference_models.models.base.types.InstancesRLEMasks`, rather
than a dense `torch.Tensor`). It is otherwise identical to
`instance_segmentation_prediction` - the same `image_metadata` and
`bboxes_metadata` conventions apply.

Example:
```
inference_models.InstanceDetections(
    xyxy=torch.Tensor([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5]]
    ),
    mask=InstancesRLEMasks(image_size=(425, 640), masks=[...]),
    confidence=torch.Tensor([    0.84955,     0.74344]),
    class_id=torch.Tensor([2, 7]),
    image_metadata={
        "class_names": {0: "car", 1: "truck"},
        "image_dimensions": [425, 640],
        "parent_id": "image.[0]",
        "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
        "prediction_type": "rle-instance-segmentation",
        "root_parent_id": "image.[0]",
        "root_parent_coordinates": [0, 0],
        "root_parent_dimensions": [425, 640],
        "parent_coordinates": [0, 0],
        "parent_dimensions": [425, 640],
        "scaling_relative_to_parent": 1.0,
        "scaling_relative_to_root_parent": 1.0,
    }
    bboxes_metadata={
        'detection_id': [
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
        ],
    }
)
```

The additional `image_metadata` / `bboxes_metadata` fields carry the same
meaning as documented for `instance_segmentation_prediction`.

**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation preserves the RLE mask representation. Entity
details: [InstanceSegmentationInferenceResponse](https://detect.roboflow.com/docs)
"""
TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="rle_instance_segmentation_prediction",
    description="Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of inference_models.InstanceDetections(...) object",
    docs=TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="inference_models.InstanceDetections",
)


TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND_DOCS = """
This kind represents single object detection prediction in form of Tuple[`inference_models.KeyPoints`, Optional[`inference_models.Detections`]] object.


Example:
```
(
    inference_models.KeyPoints(
        xy=torch.Tensor([
            [[10, 20], [10, 20], [10, 20], [10, 20]],
            [[10, 20], [10, 20], [10, 20], [10, 20]],
            [[10, 20], [10, 20], [10, 20], [10, 20]],
           [[10, 20], [10, 20], [10, 20], [10, 20]],
        ])
        confidence=torch.Tensor([
            [    0.84955,     0.74344,     0.45636,     0.86537],
            [    0.84955,     0.74344,     0.45636,     0.86537],
            [    0.84955,     0.74344,     0.45636,     0.86537],
            [    0.84955,     0.74344,     0.45636,     0.86537]   
        ]), 
        class_id=torch.Tensor([2, 7, 2, 0]), 
        image_metadata={
            "keypoints_class_names": {0: "car", 1: "truck"},
            "image_dimensions": [425, 640],
            "parent_id": "image.[0]",
            "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
            "prediction_type": "object-detection",
            "root_parent_id": "image.[0]",
            "root_parent_coordinates": [0, 0],
            "root_parent_dimensions": [425, 640],
            "parent_coordinates": [0, 0],
            "parent_dimensions": [425, 640],
            "scaling_relative_to_parent": 1.0,
            "scaling_relative_to_root_parent": 1.0,
        },
    )
    inference_models.Detections(
        xyxy=torch.Tensor([
           [        865,       153.5,        1189,       422.5],
           [      192.5,        77.5,       995.5,       722.5],
           [        194,          82,         996,         726],
           [        460,         333,         704,         389]]
        ), 
        confidence=torch.Tensor([    0.84955,     0.74344,     0.45636,     0.86537]), 
        class_id=torch.Tensor([2, 7, 2, 0]), 
        tracker_id=None, 
        image_metadata={
            "class_names": {0: "car", 1: "truck"},
            "image_dimensions": [425, 640],
            "parent_id": "image.[0]",
            "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
            "prediction_type": "object-detection",
            "root_parent_id": "image.[0]",
            "root_parent_coordinates": [0, 0],
            "root_parent_dimensions": [425, 640],
            "parent_coordinates": [0, 0],
            "parent_dimensions": [425, 640],
            "scaling_relative_to_parent": 1.0,
            "scaling_relative_to_root_parent": 1.0,
        }
        key_points_metadata={
            'detection_id': [
                '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
                '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
            ],
        }
    )
)

```

Prior to [sv.Keypoints(...)](https://supervision.roboflow.com/0.21.0/keypoint/core/) we introduced 
keypoints detection based on [`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object.
The decision was suboptimal so we would need to revert in the future, but for now this is the format of
data for keypoints detection. 

The design of metadata is also suboptimal (as metadata regarding whole image is duplicated across all 
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

* `keypoints_class_name` array of variable size 1D arrays of string with key points class names

* `keypoints_class_id` array of variable size 1D arrays of int with key points class ids

* `keypoints_confidence` array of variable size 1D arrays of float with key points confidence

* `keypoints_xy` array of variable size 2D arrays of coordinates of keypoints in `(x, y)` format

**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output, but **loosing keypoints details** - which can be recovered if output 
JSON field is parsed. Entity details: [KeypointsDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND = Kind(
    name="keypoint_detection_prediction",
    description="Prediction with detected bounding boxes and detected keypoints in form of a `(inference_models.KeyPoints, Optional[inference_models.Detections])` tuple",
    docs=TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="Tuple[inference_models.KeyPoints, Optional[inference_models.Detections]]",
)


TENSOR_NATIVE_QR_CODE_DETECTION_KIND_DOCS = """
This kind represents batch of predictions regarding QR codes location and data their provide.

Example:
```
inference_models.Detections(
    xyxy=torch.Tensor([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    confidence=torch.Tensor([    0.84955,     0.74344,     0.45636,     0.86537]), 
    class_id=torch.Tensor([2, 7, 2, 0]), 
    tracker_id=None, 
    image_metadata={
        "class_names": {0: "qr-code"},
        "image_dimensions": [425, 640],
        "parent_id": "image.[0]",
        "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
        "prediction_type": "qrcode-detection",
        "root_parent_id": "image.[0]",
        "root_parent_coordinates": [0, 0],
        "root_parent_dimensions": [425, 640],
        "parent_coordinates": [0, 0],
        "parent_dimensions": [425, 640],
        "scaling_relative_to_parent": 1.0,
        "scaling_relative_to_root_parent": 1.0,
    }
    bboxes_metadata={
        'detection_id': [
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ],
        "qr_codes": ["a", "b", "c", "d"]
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

* `qr_codes` - extracted QR code

**SERIALISATION:**
Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [ObjectDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
TENSOR_NATIVE_QR_CODE_DETECTION_KIND = Kind(
    name="qr_code_detection",
    description="Prediction with QR code detection",
    docs=TENSOR_NATIVE_QR_CODE_DETECTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="inference_models.Detections",
)


TENSOR_NATIVE_BAR_CODE_DETECTION_KIND_DOCS = """
This kind represents batch of predictions regarding barcodes location and data their provide.

Example:
```
inference_models.Detections(
    xyxy=torch.Tensor([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    confidence=torch.Tensor([    0.84955,     0.74344,     0.45636,     0.86537]), 
    class_id=torch.Tensor([2, 7, 2, 0]), 
    tracker_id=None, 
    image_metadata={
        "class_names": {0: "barcode"},
        "image_dimensions": [425, 640],
        "parent_id": "image.[0]",
        "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
        "prediction_type": "barcode-detection",
        "root_parent_id": "image.[0]",
        "root_parent_coordinates": [0, 0],
        "root_parent_dimensions": [425, 640],
        "parent_coordinates": [0, 0],
        "parent_dimensions": [425, 640],
        "scaling_relative_to_parent": 1.0,
        "scaling_relative_to_root_parent": 1.0,
    }
    bboxes_metadata={
        'detection_id': [
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ],
        "barcodes": ["a", "b", "c", "d"]
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

* `barcodes` - extracted barcode

**SERIALISATION:**
Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [ObjectDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
TENSOR_NATIVE_BAR_CODE_DETECTION_KIND = Kind(
    name="bar_code_detection",
    description="Prediction with barcode detection",
    docs=TENSOR_NATIVE_BAR_CODE_DETECTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="inference_models.Detections",
)


TENSOR_NATIVE_SEMANTIC_SEGMENTATION_PREDICTION_KIND_DOCS = """
This kind represents a single semantic segmentation prediction in form of an
`inference_models.InstanceDetections` object with one detection per predicted class. Each
detection carries an RLE-encoded mask covering all pixels assigned to that class.

**Why RLE and not polygons:**

Semantic segmentation assigns a class label to every pixel in the image. A single class
can appear in multiple spatially disconnected regions (e.g., two separate "person" regions
on opposite sides of the frame). Polygon-based serialization uses `cv2.findContours()`,
which only retains the first contiguous contour and silently discards all others — causing
irreversible data loss for non-contiguous masks. RLE (Run-Length Encoding, COCO standard)
is a pixel-level encoding that represents the complete mask regardless of spatial topology,
making it the only correct serialization format for semantic segmentation masks.

**Internal representation:** an `inference_models.InstanceDetections` carrier (the same
per-class RLE carrier used by `rle_instance_segmentation_prediction`), with one row per
predicted class:
- `xyxy` — `torch.Tensor` of tight bounding boxes enclosing all pixels of each class
- `mask` — `inference_models.models.base.types.InstancesRLEMasks` holding one COCO RLE
  entry per class (NOT a dense `torch.Tensor`, NO polygon collapse)
- `class_id` — `torch.Tensor` of integer class IDs
- `confidence` — `torch.Tensor` of mean confidence over all pixels of each class
- `image_metadata[CLASS_NAMES_KEY]` — class-id -> class-name mapping for the prediction
- `bboxes_metadata` — same per-box metadata conventions as `instance_segmentation_prediction`
  (e.g. `detection_id`); see that kind for the full key list.

Example:
```
inference_models.InstanceDetections(
    xyxy=torch.Tensor([
       [          0,           0,         640,         480],
       [        100,         120,         400,         360]]
    ),
    mask=InstancesRLEMasks(image_size=(480, 640), masks=[...]),
    confidence=torch.Tensor([    0.97,     0.92]),
    class_id=torch.Tensor([0, 1]),
    image_metadata={
        "class_names": {0: "background", 1: "person"},
        "image_dimensions": [480, 640],
        "parent_id": "image.[0]",
        "inference_id": "51dfa8d5-261c-4dcb-ab30-9aafe9b52379",
        "prediction_type": "semantic-segmentation",
        "root_parent_id": "image.[0]",
        "root_parent_coordinates": [0, 0],
        "root_parent_dimensions": [480, 640],
        "parent_coordinates": [0, 0],
        "parent_dimensions": [480, 640],
        "scaling_relative_to_parent": 1.0,
        "scaling_relative_to_root_parent": 1.0,
    }
    bboxes_metadata={
        'detection_id': [
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
        ],
    }
)
```

**Serialised format** (one entry per class in `predictions`):

```json
{
    "image": {"width": 640, "height": 480},
    "predictions": [
        {
            "x": 320.0, "y": 240.0, "width": 200.0, "height": 180.0,
            "confidence": 0.92,
            "class_id": 1,
            "class": "person",
            "detection_id": "a1b2c3d4-...",
            "rle_mask": {"size": [480, 640], "counts": "XYZ..."}
        }
    ]
}
```

**Decoding RLE masks:**

```python
import pycocotools.mask as mask_utils
import numpy as np

rle = prediction["rle_mask"]
binary_mask = mask_utils.decode(rle).astype(bool)  # shape: (H, W)
```
"""
TENSOR_NATIVE_SEMANTIC_SEGMENTATION_PREDICTION_KIND = Kind(
    name="semantic_segmentation_prediction",
    description="Prediction with per-pixel class label and confidence for semantic segmentation",
    docs=TENSOR_NATIVE_SEMANTIC_SEGMENTATION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="inference_models.InstanceDetections",
)


TENSOR_KIND_DOCS = """
This kind represents a raw, multi-dimensional numeric tensor kept on-device as a
`torch.Tensor`. It is the tensor-native counterpart of `numpy_array`, but it is a
standalone kind with its own name (`tensor`) - it does NOT replace or alias
`numpy_array`; the two coexist.

It is used by blocks that emit dense numeric maps (for example, the depth map
produced by Depth Estimation) and want to keep the data on the accelerator,
avoiding an eager GPU -> CPU / numpy round-trip.
"""
TENSOR_KIND = Kind(
    name="tensor",
    description="A raw multi-dimensional numeric tensor (torch.Tensor)",
    docs=TENSOR_KIND_DOCS,
    serialised_data_type="list",
    internal_data_type="torch.Tensor",
)
