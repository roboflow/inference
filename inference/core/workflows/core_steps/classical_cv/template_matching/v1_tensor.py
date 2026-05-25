from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.classical_cv.template_matching.v1 import (
    TemplateMatchingBlockV1 as _NumpyImpl,
)

TemplateMatchingBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
