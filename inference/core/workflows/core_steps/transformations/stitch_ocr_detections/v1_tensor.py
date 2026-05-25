from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v1 import (
    StitchOCRDetectionsBlockV1 as _NumpyImpl,
)

StitchOCRDetectionsBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
