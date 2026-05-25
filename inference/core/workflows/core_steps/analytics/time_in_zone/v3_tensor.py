from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v3 import (
    TimeInZoneBlockV3 as _NumpyImpl,
)

TimeInZoneBlockV3 = make_tensor_wrapper_block(_NumpyImpl)
