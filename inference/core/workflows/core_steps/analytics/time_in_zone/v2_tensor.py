from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v2 import (
    TimeInZoneBlockV2 as _NumpyImpl,
)

TimeInZoneBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
