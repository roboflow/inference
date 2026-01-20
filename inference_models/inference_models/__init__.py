"""
This is a definition of public interface of the `inference-models` package.
We want the clients code to only explicitly import from this module, to improve stability and
make it clear what is the intended public API.

Whenever using `inference-models` in your code, import only from this module, like this:
```python
from inference_models import AutoModel
```

We keep the list of exposed symbols small, to make it easier to reason about the API and to avoid
polluting the namespace. If you see a symbol that you need but is not exposed here, please open an issue
https://github.com/roboflow/inference/issues/new
"""
import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from inference_models.entities import ColorFormat
from inference_models.model_pipelines.auto_loaders.core import AutoModelPipeline
from inference_models.models.auto_loaders.core import AutoModel
from inference_models.models.auto_loaders.entities import AnyModel, BackendType
from inference_models.models.base.classification import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.depth_estimation import DepthEstimationModel
from inference_models.models.base.documents_parsing import (
    StructuredOCRModel,
    TextOnlyOCRModel,
)
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints,
    KeyPointsDetectionModel,
)
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
    OpenVocabularyObjectDetectionModel,
)
from inference_models.models.base.semantic_segmentation import SemanticSegmentationModel
from inference_models.weights_providers.entities import Quantization
