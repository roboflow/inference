from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2 import (
    RoboflowDatasetUploadBlockV2 as _NumpyImpl,
)

RoboflowDatasetUploadBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
