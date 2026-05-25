from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.per_class_confidence_filter.v1 import (
    PerClassConfidenceFilterBlockV1 as _NumpyImpl,
)

PerClassConfidenceFilterBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
