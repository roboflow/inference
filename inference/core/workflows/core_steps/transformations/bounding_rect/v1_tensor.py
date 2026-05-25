from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.bounding_rect.v1 import (
    BoundingRectBlockV1 as _NumpyImpl,
)

BoundingRectBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
