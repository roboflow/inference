from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.fusion.detections_consensus.v1 import (
    DetectionsConsensusBlockV1 as _NumpyImpl,
)

DetectionsConsensusBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
