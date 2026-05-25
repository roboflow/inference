from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v2 import (
    StitchOCRDetectionsBlockV2 as _NumpyImpl,
)

StitchOCRDetectionsBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
