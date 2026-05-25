from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v2 import (
    VLMAsClassifierBlockV2 as _NumpyImpl,
)

VLMAsClassifierBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
