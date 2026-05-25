from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
    DynamicZonesBlockV1 as _NumpyImpl,
)

DynamicZonesBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
