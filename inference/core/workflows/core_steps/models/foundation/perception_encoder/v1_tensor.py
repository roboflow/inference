from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.perception_encoder.v1 import (
    PerceptionEncoderModelBlockV1 as _NumpyImpl,
)

PerceptionEncoderModelBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
