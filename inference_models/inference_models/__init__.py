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

import _thread
import os
import sys
import tempfile

_OFFLINE_MODE_PROCESS_LATCH_ENV = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
_OFFLINE_MODE_PROCESS_STATE_MODULE = "_roboflow_inference_process_state"

_offline_mode_process_state = sys.modules.get(_OFFLINE_MODE_PROCESS_STATE_MODULE)
_offline_mode_initialization_lock_owned = False
if _offline_mode_process_state is None:
    _candidate_process_state = type(sys)(_OFFLINE_MODE_PROCESS_STATE_MODULE)
    _candidate_process_state.lock = _thread.RLock()
    _candidate_process_state.lock.acquire()
    _offline_mode_process_state = sys.modules.setdefault(
        _OFFLINE_MODE_PROCESS_STATE_MODULE,
        _candidate_process_state,
    )
    if _offline_mode_process_state is _candidate_process_state:
        _offline_mode_initialization_lock_owned = True
    else:
        _candidate_process_state.lock.release()

if not _offline_mode_initialization_lock_owned:
    _offline_mode_initialization_lock = _offline_mode_process_state.__dict__.setdefault(
        "lock", _thread.RLock()
    )
    _offline_mode_initialization_lock.acquire()
else:
    _offline_mode_initialization_lock = _offline_mode_process_state.lock

try:
    if not hasattr(_offline_mode_process_state, "offline_mode") and hasattr(
        _offline_mode_process_state, "offline_mode_initialization_error"
    ):
        raise _offline_mode_process_state.offline_mode_initialization_error

    if hasattr(os, "register_at_fork") and not getattr(
        _offline_mode_process_state, "at_fork_registered", False
    ):

        def _reset_offline_mode_lock_after_fork() -> None:
            _offline_mode_process_state.lock = _thread.RLock()

        os.register_at_fork(after_in_child=_reset_offline_mode_lock_after_fork)
        _offline_mode_process_state.at_fork_registered = True

    if not hasattr(_offline_mode_process_state, "offline_mode_startup_snapshot"):
        _offline_mode_process_state.offline_mode_startup_snapshot = (
            os.environ.get("OFFLINE_MODE"),
            os.environ.get(_OFFLINE_MODE_PROCESS_LATCH_ENV),
        )
    (
        _offline_mode_from_process_environment,
        _inherited_offline_mode_at_import,
    ) = _offline_mode_process_state.offline_mode_startup_snapshot

    _offline_mode_declared_at_startup = (
        _offline_mode_from_process_environment is not None
    )
    _dotenv_path = os.path.join(os.getcwd(), ".env")
    from dotenv import dotenv_values, load_dotenv

    if (
        not hasattr(_offline_mode_process_state, "offline_mode")
        and not _offline_mode_declared_at_startup
        and _inherited_offline_mode_at_import is None
    ):
        # `inference` has historically loaded the project-local .env file. Both
        # packages must resolve this security-sensitive startup flag identically
        # so import order cannot silently select online mode.
        _dotenv_values = dotenv_values(_dotenv_path)
        _offline_mode_declared_at_startup = "OFFLINE_MODE" in _dotenv_values
        if _offline_mode_declared_at_startup:
            _offline_mode_from_process_environment = _dotenv_values.get("OFFLINE_MODE")

    # Load the rest of the project-local configuration before importing any
    # module that snapshots environment variables. The raw OFFLINE_MODE value
    # above remains authoritative so a concurrent mutation cannot change the
    # process latch.
    load_dotenv(_dotenv_path)

    # Keep implicit Hugging Face downloads inside the cache volume that a
    # fresh offline process will mount. The Hub and Transformers snapshot
    # these variables at import time, so publish every relevant path before
    # importing either library. Explicit user configuration always wins.
    _dependency_cache_root = (
        os.environ.get("INFERENCE_HOME")
        or os.environ.get("MODEL_CACHE_DIR")
        or "/tmp/cache"
    )
    os.environ.setdefault(
        "HF_HOME",
        os.path.join(_dependency_cache_root, "hf_home"),
    )
    os.environ.setdefault(
        "HF_HUB_CACHE",
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
        or os.path.join(os.environ["HF_HOME"], "hub"),
    )
    os.environ.setdefault(
        "HF_MODULES_CACHE",
        os.path.join(os.environ["HF_HOME"], "modules"),
    )
    # Ultralytics persists settings even while merely loading them. Keep this
    # mutable runtime state out of the model artifact volume, which is expected
    # to be mounted read-only during an offline restart.
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(
            tempfile.gettempdir(),
            "roboflow-inference",
            "ultralytics",
        ),
    )

    from inference_models._offline import publish_offline_mode

    OFFLINE_MODE = publish_offline_mode(
        offline_mode_process_state=_offline_mode_process_state,
        requested_offline_mode=_offline_mode_from_process_environment,
        inherited_offline_mode=_inherited_offline_mode_at_import,
        requested_offline_mode_declared=_offline_mode_declared_at_startup,
    )
    if OFFLINE_MODE and any(
        module_name == "ultralytics" or module_name.startswith("ultralytics.")
        for module_name in sys.modules
    ):
        raise RuntimeError(
            "Ultralytics was imported before inference-models could establish "
            "OFFLINE_MODE. Restart the process and import inference_models first."
        )
except Exception as error:
    if not hasattr(_offline_mode_process_state, "offline_mode"):
        _offline_mode_process_state.offline_mode_initialization_error = error
    raise
finally:
    _offline_mode_initialization_lock.release()

import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from inference_models.entities import ColorFormat, Confidence
from inference_models.model_pipelines.auto_loaders.core import AutoModelPipeline
from inference_models.models.auto_loaders.core import (
    AutoModel,
    find_cached_model_package_dir,
)
from inference_models.models.auto_loaders.entities import (
    AnyModel,
    BackendType,
    PreProcessingOverrides,
)
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
    InstanceSegmentationMaskFormat,
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
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.weights_providers.entities import Quantization
