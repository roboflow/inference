from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2 import (
    VLMAsDetectorBlockV2 as _NumpyImpl,
)

VLMAsDetectorBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
