from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v1 import (
    TimeInZoneBlockV1 as _NumpyImpl,
)

TimeInZoneBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
