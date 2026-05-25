from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1 import (
    RoboflowDatasetUploadBlockV1 as _NumpyImpl,
)

RoboflowDatasetUploadBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
