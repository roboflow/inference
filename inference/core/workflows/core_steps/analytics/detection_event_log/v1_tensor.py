from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.detection_event_log.v1 import (
    DetectionEventLogBlockV1 as _NumpyImpl,
)

DetectionEventLogBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
