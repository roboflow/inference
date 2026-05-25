from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.easy_ocr.v1 import (
    EasyOCRBlockV1 as _NumpyImpl,
)

EasyOCRBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
