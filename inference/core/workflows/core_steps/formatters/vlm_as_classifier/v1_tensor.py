from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v1 import (
    VLMAsClassifierBlockV1 as _NumpyImpl,
)

VLMAsClassifierBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
