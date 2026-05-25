from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.path_deviation.v1 import (
    PathDeviationAnalyticsBlockV1 as _NumpyImpl,
)

PathDeviationAnalyticsBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
