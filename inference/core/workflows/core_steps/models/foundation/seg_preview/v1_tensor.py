from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.seg_preview.v1 import (
    SegPreviewBlockV1 as _NumpyImpl,
)

SegPreviewBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
