from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.path_deviation.v2 import (
    PathDeviationAnalyticsBlockV2 as _NumpyImpl,
)

PathDeviationAnalyticsBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
