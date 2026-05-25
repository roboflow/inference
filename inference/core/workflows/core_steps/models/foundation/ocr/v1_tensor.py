from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.ocr.v1 import (
    OCRModelBlockV1 as _NumpyImpl,
)

OCRModelBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
