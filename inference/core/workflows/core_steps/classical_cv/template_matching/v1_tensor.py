from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Locate instances of a given template within a specified image."
)
LONG_DESCRIPTION: str = """
Locate instances of a template image within a larger image using template matching with normalized cross-correlation, finding exact or near-exact matches of the template pattern at any location in the image, outputting bounding box detections with optional NMS filtering for object detection, logo detection, pattern recognition, and template-based object localization workflows.

## How This Block Works

This block searches for occurrences of a template image within a larger input image using normalized cross-correlation template matching. The block:

1. Receives an input image and a template image (smaller pattern to search for)
2. Converts both images to grayscale for template matching (template matching typically works on grayscale images for efficiency and robustness)
3. Performs template matching using OpenCV's matchTemplate with TM_CCOEFF_NORMED method:
   - Slides the template across the input image at every possible position
   - Computes normalized cross-correlation coefficient at each position (measures similarity between template and image region)
   - Generates a similarity map showing how well the template matches at each location
4. Identifies match locations where similarity exceeds the matching_threshold:
   - Finds all positions where the correlation coefficient is greater than or equal to the threshold
   - Threshold values range from 0.0 to 1.0, with higher values requiring closer matches
   - Lower thresholds find more potential matches (including partial matches), higher thresholds find only very similar matches
5. Creates bounding boxes for each match:
   - Each match location becomes a detection with a bounding box matching the template's dimensions
   - All detections have confidence of 1.0 (they met the threshold requirement)
   - All detections are assigned class "template_match" and class_id 0
   - Each detection gets a unique detection ID for tracking
6. Optionally applies Non-Maximum Suppression (NMS) to filter overlapping detections:
   - Template matching often produces many overlapping detections at the same location (duplicate matches)
   - NMS removes overlapping detections, keeping only the best match in each area
   - NMS threshold controls how much overlap is allowed before removing detections
   - Can be disabled (apply_nms=False) if NMS becomes computationally intractable with very large numbers of matches
7. Attaches metadata to detections:
   - Sets parent_id to reference the input image
   - Sets prediction_type to "object-detection"
   - Stores image dimensions for coordinate reference
   - Attaches parent coordinate information for workflow tracking
8. Returns detection predictions in sv.Detections format along with the total number of matches found

The block uses normalized cross-correlation which is effective for finding exact or near-exact template matches. It works best when the template appears in the image at the same scale, rotation, and lighting conditions. The method tends to produce many overlapping detections for the same match location, which is why NMS filtering is important. However, in cases with extremely large numbers of matches (e.g., repeating patterns), NMS may become computationally expensive and can be disabled if needed.

## Common Use Cases

- **Logo and Brand Detection**: Find specific logos or brand elements within images (e.g., detect company logos in photos, find brand markers in images, locate specific logo patterns in scenes), enabling logo detection workflows
- **Exact Pattern Matching**: Locate specific patterns or objects that appear identically in images (e.g., find specific UI elements in screenshots, detect exact patterns in images, locate specific visual elements), enabling exact pattern detection workflows
- **Quality Control and Inspection**: Find reference patterns or features for quality inspection (e.g., detect specific features in manufacturing images, find reference markers for alignment, locate inspection targets), enabling quality control workflows
- **Object Localization**: Locate specific objects or regions when exact appearance is known (e.g., find specific objects with known appearance, locate reference objects in images, detect specific visual elements), enabling template-based object localization
- **Document Processing**: Find specific elements or regions in documents (e.g., locate form fields in documents, detect specific document elements, find reference markers in scanned documents), enabling document processing workflows
- **UI Element Detection**: Detect specific UI components or elements in interface images (e.g., find buttons in UI screenshots, locate specific UI elements, detect interface components), enabling UI analysis workflows

## Connecting to Other Blocks

This block receives an image and template, and produces detection predictions:

- **After image input blocks** to find template patterns in input images (e.g., search for templates in input images, locate patterns in camera feeds, find templates in image streams), enabling template matching workflows
- **After preprocessing blocks** to find templates in preprocessed images (e.g., match templates after image enhancement, find patterns in filtered images, locate templates in normalized images), enabling preprocessed template matching
- **Before visualization blocks** to visualize template match locations (e.g., visualize detected template matches, display bounding boxes for matches, show template match results), enabling template match visualization workflows
- **Before filtering blocks** to filter template matches by criteria (e.g., filter matches by location, select specific match regions, refine template match results), enabling filtered template matching workflows
- **Before crop blocks** to extract regions around template matches (e.g., crop areas around matches, extract match regions for analysis, crop template match locations), enabling template-based region extraction
- **In quality control workflows** where template matching is used for inspection or alignment (e.g., find reference markers for alignment, detect inspection targets, locate quality control features), enabling quality control template matching workflows
"""


class TemplateMatchingManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/template_matching@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Template Matching",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-crosshairs",
                "blockPriority": 0.5,
                "opencv": True,
            },
        }
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Large image in which to search for the template pattern. The template will be searched across this entire image at all possible positions. The image is converted to grayscale internally for template matching. Template matching works best when the image and template have similar lighting conditions and the template appears at similar scale and orientation in the image.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    template: Selector(kind=[IMAGE_KIND]) = Field(
        title="Template Image",
        description="Small template image pattern to search for within the input image. The template should be smaller than the input image. The template is converted to grayscale internally for matching. Template matching finds exact or near-exact matches of this template at any location in the input image. Works best when the template appears in the image at the same scale, rotation, and lighting conditions. The template's dimensions determine the size of the detection bounding boxes.",
        examples=["$inputs.template", "$steps.cropping.template"],
        validation_alias=AliasChoices("template", "templates"),
    )
    matching_threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Matching Threshold",
        description="Minimum similarity threshold (0.0 to 1.0) required for a template match. Higher values (closer to 1.0) require very close matches and find fewer but more precise matches. Lower values (closer to 0.0) allow more lenient matches and find more potential matches including partial matches. Default is 0.8, which requires fairly close matches. Use lower thresholds (0.6-0.7) to find more matches or handle slight variations. Use higher thresholds (0.85-0.95) for exact matches only. The threshold compares normalized cross-correlation coefficients from template matching.",
        default=0.8,
        examples=[0.8, "$inputs.threshold"],
    )
    apply_nms: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        title="Apply NMS",
        description="Whether to apply Non-Maximum Suppression (NMS) to filter overlapping detections. Template matching often produces many overlapping detections at the same location. NMS removes overlapping detections, keeping only the best match in each area. Default is True (recommended for most cases). Set to False if: (1) the number of matches is extremely large (NMS may become computationally expensive), (2) you want to see all raw matches without filtering, or (3) matches are intentionally close together and should all be kept. When disabled, you may see many duplicate detections for the same match location.",
        default=True,
        examples=["$inputs.apply_nms", False],
    )
    nms_threshold: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), FloatZeroToOne] = (
        Field(
            title="NMS threshold",
            description="Intersection over Union (IoU) threshold for Non-Maximum Suppression. Only relevant when apply_nms is True. Detections with IoU overlap greater than this threshold are considered duplicates, and only the detection with highest confidence is kept. Lower values (0.3-0.4) are more aggressive at removing overlaps, removing detections that are only slightly overlapping. Higher values (0.6-0.7) are more lenient, only removing heavily overlapping detections. Default is 0.5, which provides balanced overlap filtering. Adjust based on how much overlap you expect between template matches and how close together valid matches can be.",
            default=0.5,
            examples=["$inputs.nms_threshold", 0.3],
        )
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(
                name="number_of_matches",
                kind=[INTEGER_KIND],
            ),
        ]


class TemplateMatchingBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[TemplateMatchingManifest]:
        return TemplateMatchingManifest

    def run(
        self,
        image: WorkflowImageData,
        template: WorkflowImageData,
        matching_threshold: float,
        apply_nms: bool,
        nms_threshold: float,
    ) -> BlockResult:
        detections = apply_template_matching(
            image=image,
            template=template.numpy_image,
            matching_threshold=matching_threshold,
            apply_nms=apply_nms,
            nms_threshold=nms_threshold,
        )
        return {"predictions": detections, "number_of_matches": len(detections)}


def apply_template_matching(
    image: WorkflowImageData,
    template: np.ndarray,
    matching_threshold: float,
    apply_nms: bool,
    nms_threshold: float,
) -> sv.Detections:
    img_gray = cv2.cvtColor(image.numpy_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= matching_threshold)
    xyxy, confidence, class_id, class_name, detections_id = [], [], [], [], []
    for pt in zip(*loc[::-1]):
        top_left = pt
        bottom_right = (pt[0] + w, pt[1] + h)
        xyxy.append(top_left + bottom_right)
        confidence.append(1.0)
        class_id.append(0)
        class_name.append("template_match")
        detections_id.append(str(uuid4()))
    if len(xyxy) == 0:
        return sv.Detections.empty()
    detections = sv.Detections(
        xyxy=np.array(xyxy).astype(np.int32),
        confidence=np.array(confidence),
        class_id=np.array(class_id).astype(np.uint32),
        data={CLASS_NAME_DATA_FIELD: np.array(class_name)},
    )
    if apply_nms:
        detections = detections.with_nms(threshold=nms_threshold)
    detections[PARENT_ID_KEY] = np.array(
        [image.parent_metadata.parent_id] * len(detections)
    )
    detections[PREDICTION_TYPE_KEY] = np.array(["object-detection"] * len(detections))
    detections[DETECTION_ID_KEY] = np.array(detections_id)
    image_height, image_width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.array(
        [[image_height, image_width]] * len(detections)
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )
