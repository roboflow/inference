"""Shared model-block algorithms: model selection, option binding and results.

A step's model and options are one value for the whole image batch of a call;
a selector may change them between calls, not between images. A phased
task-model block runs::

    network_input   check images and options, resolve the call's model,
                    bind its options, pre_process all images  -> ModelInput
    raw_output      model.forward(batch)
    result          model.post_process(raw, metadata, **options), then one
                    result per image, in batch order

Each call builds its own ``ModelInput``; the block instance holds only the
model cache. Overlapping phased calls may share one model, so each model must
support the calls it receives and complete its outputs before returning them.
An empty batch resolves no model, and every argument check (``one_per_call``,
``scalar_options``, ``checked_class_filter``) runs before the model loads.

Options reach a model only through its ``post_process`` keywords
(``post_process_options``). An unset option gets its V1 default only when the
model names that keyword; a set option the model does not read fails the step
before any model runs. A confidence mode (``"best"`` / ``"default"``) is not a
set value. Keywords collected by ``**kwargs`` do not count, and
``IGNORED_KEYWORDS`` lists backends that name a keyword but never use it; both
facts come from a source survey of the ``inference_models`` task backends.
"""

import difflib
import inspect
import uuid
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.row_identity import with_row_identity

EMPTY_CLASS_FILTER_MESSAGE = (
    "class_filter is an empty list, which would keep no class; omit "
    "class_filter or set it to null to keep every class"
)


# post_process keywords a backend names but never uses, by the qualified name
# of the post_process that names them. Setting one fails like a missing one.
IGNORED_KEYWORDS: Mapping[str, frozenset] = MappingProxyType(
    {
        "inference_models.models.yolonas.yolonas_object_detection_trt."
        "YOLONasForObjectDetectionTRT.post_process": frozenset({"class_agnostic_nms"}),
    }
)

# V1 defaults, passed to models that read them when the step leaves them unset.
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_DETECTIONS = 300
DEFAULT_CLASS_AGNOSTIC_NMS = False


class ModelOption(NamedTuple):
    """One step option and the ``post_process`` keyword that carries it.

    Args:
        param: Step parameter name, used in error messages.
        keyword: ``post_process`` keyword argument.
        value: Value set by the step; ``None`` means not set.
        default: Value passed when not set; ``None`` passes nothing.
    """

    param: str
    keyword: str
    value: Any
    default: Any = None


class ModelInput(NamedTuple):
    """What ``network_input`` hands the later phases of one call.

    Args:
        request: The call's model request, for errors and the ``model_id``
            output.
        model: The resolved model; the cache owns it.
        batch: ``pre_process`` output for all images of the call.
        metadata: ``pre_process`` metadata (geometry), passed unchanged to
            ``post_process``; ``None`` for models that return none.
        options: ``post_process`` keywords bound for this model.
        class_ids: Ids ``class_filter`` keeps; ``None`` keeps every class.
    """

    request: ModelRequest
    model: Any
    batch: Any
    metadata: Any
    options: Mapping[str, Any]
    class_ids: Optional[Tuple[int, ...]] = None


def confidence_option(confidence: Union[float, str]) -> ModelOption:
    """The ``confidence`` option: a number is set, a mode is a default.

    A model without a ``confidence`` keyword applies its own threshold, which
    is what both modes ask for; a number it cannot read fails the step.
    """
    if isinstance(confidence, str):
        return ModelOption("confidence", "confidence", None, confidence)

    option = ModelOption("confidence", "confidence", confidence)

    return option


def nms_options(
    *,
    confidence: Union[float, str],
    iou_threshold: Optional[float],
    max_detections: Optional[int],
    class_agnostic_nms: Optional[bool],
) -> Tuple[ModelOption, ...]:
    """The ``post_process`` options of a box-producing step."""
    options = (
        confidence_option(confidence),
        ModelOption(
            "iou_threshold", "iou_threshold", iou_threshold, DEFAULT_IOU_THRESHOLD
        ),
        ModelOption(
            "max_detections", "max_detections", max_detections, DEFAULT_MAX_DETECTIONS
        ),
        ModelOption(
            "class_agnostic_nms",
            "class_agnostic_nms",
            class_agnostic_nms,
            DEFAULT_CLASS_AGNOSTIC_NMS,
        ),
    )

    return options


def models_cache(
    models: Union[Mapping[Any, Any], ModelCache], *, block: str
) -> ModelCache:
    """Wrap the session's ``models`` resource in a ``ModelCache``.

    A mapping is copied (not its models) and serves only what it holds: a
    later change of the host's dictionary does not change a running block.

    Args:
        models: A ``ModelCache``, or a mapping of model id (or
            ``ModelRequest``) to a loaded model.
        block: Block type, for error messages.

    Returns:
        The cache the block reads models from.

    Raises:
        TypeError: When ``models`` is neither.
    """
    if isinstance(models, ModelCache):
        return models

    if not isinstance(models, Mapping):
        raise TypeError(
            f"{block} needs the 'models' resource as a mapping of model "
            f"id to loaded model, or a ModelCache, got {type(models).__name__}"
        )

    # The snapshot keeps the model objects; nothing is loaded.
    supplied = MappingProxyType(dict(models))
    cache = ModelCache(_refuse_unsupplied(supplied), supplied=supplied)

    return cache


def one_per_call(value: Any, *, name: str, block: str) -> None:
    """Refuse a ``Batch`` of per-image values for a one-per-call field.

    Compiled workflows never deliver a ``Batch`` to a field that is not
    batch-delivered; this names the mistake for direct ``run()`` callers
    instead of reading one image's value.

    Args:
        value: The value of one configuration field for the whole call.
        name: Field name, for the error message.
        block: Block type, for the error message.

    Raises:
        ValueError: When ``value`` is a ``Batch``.
    """
    if isinstance(value, Batch):
        raise ValueError(
            f"{block}: {name} is one value for every image of a call, got a "
            f"Batch of {len(value)} values; run the step once per value"
        )


def model_request(
    model_id: str, model_package_id: Optional[str], *, block: str
) -> ModelRequest:
    """The one model request of a call.

    Args:
        model_id: Model id for every image of the call.
        model_package_id: Package id for every image, or ``None``.
        block: Block type, for error messages.

    Returns:
        The request to resolve in the model cache.

    Raises:
        ValueError: On a ``Batch`` of ids (one value per image is not
            supported), an empty or non-text id.
    """
    one_per_call(model_id, name="model_id", block=block)
    one_per_call(model_package_id, name="model_package_id", block=block)
    try:
        request = ModelRequest(model_id, model_package_id)
    except ValueError as error:
        raise ValueError(f"{block}: {error}") from error

    return request


def rgb_tensors(image: Batch, *, block: str) -> List[torch.Tensor]:
    """The ``(3, H, W)`` tensors of a batch of RGB images, in batch order.

    Args:
        image: The batch of images of one call.
        block: Block type, for error messages.

    Returns:
        One tensor per image, as ``pre_process`` takes them.

    Raises:
        ValueError: When an image does not have 3 channels.
    """
    for member in image:
        require_rgb(member, block=block)
    tensors = [member.tensor_image for member in image]

    return tensors


def post_process_options(
    post_process: Callable[..., Any],
    options: Sequence[ModelOption],
    *,
    block: str,
    request: ModelRequest,
) -> Dict[str, Any]:
    """Bind step options to the keywords one model's ``post_process`` names.

    Args:
        post_process: The model's bound ``post_process``.
        options: Step options.
        block: Block type, for error messages.
        request: The model's request, for error messages.

    Returns:
        Keyword arguments for ``post_process``.

    Raises:
        ValueError: When the step sets an option the model does not read.
    """
    named = _named_parameters(post_process) - _ignored_keywords(post_process)
    keywords = {}
    for option in options:
        if option.keyword in named:
            value = option.default if option.value is None else option.value
            if value is not None:
                keywords[option.keyword] = value
        elif option.value is not None:
            raise ValueError(
                _unread_option_message(
                    option, named, options, post_process, block=block, request=request
                )
            )

    return keywords


def checked_count(
    predictions: Any, *, image_count: int, request: ModelRequest, block: str
) -> None:
    """Fail when a model returns another number of results than images."""
    if len(predictions) != image_count:
        raise ValueError(
            f"{block}: post_process of model {str(request)!r} returned "
            f"{len(predictions)} predictions for {image_count} images"
        )


def checked_confidence(
    confidence: Union[float, str], *, modes: Tuple[str, ...], block: str
) -> Union[float, str]:
    """Check a literal or selected confidence: a number 0..1 or a mode.

    Args:
        confidence: The step's confidence.
        modes: Named modes this block accepts, e.g. ``("best", "default")``.
        block: Block type, for error messages.

    Returns:
        ``confidence`` as a float, or the mode string unchanged.

    Raises:
        ValueError: On a ``Batch``, another string or a number outside 0..1.
    """
    one_per_call(confidence, name="confidence", block=block)
    if isinstance(confidence, str):
        if confidence not in modes:
            raise ValueError(
                f"{block}: confidence must be a number 0..1 or one of "
                f"{list(modes)}, got {confidence!r}"
            )
        return confidence

    if isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError(
            f"{block}: confidence must be a number 0..1 or one of {list(modes)}, "
            f"got {confidence!r}"
        )
    value = float(confidence)

    return value


def scalar_options(
    options: Sequence[ModelOption], *, block: str
) -> Sequence[ModelOption]:
    """Return ``options`` unless one holds a ``Batch`` of per-image values.

    Call it before resolving the model, so a malformed direct call loads
    nothing.

    Args:
        options: Step options of one call.
        block: Block type, for error messages.

    Returns:
        ``options`` unchanged.

    Raises:
        ValueError: When an option value is a ``Batch``.
    """
    for option in options:
        one_per_call(option.value, name=option.param, block=block)

    return options


def checked_class_filter(
    class_filter: Any, *, block: str
) -> Optional[Tuple[str, ...]]:
    """Check the structure of a step's ``class_filter`` before any model loads.

    A selected filter skipped the literal validators, so it is checked here
    the same way. Class names are matched to a model by ``class_filter_ids``.

    Args:
        class_filter: Class names to keep for every image, or ``None``.
        block: Block type, for error messages.

    Returns:
        The names, or ``None`` to keep every class.

    Raises:
        ValueError: On a ``Batch``, or a value that is not a non-empty list of
            non-empty names.
    """
    if class_filter is None:
        return None

    one_per_call(class_filter, name="class_filter", block=block)
    if isinstance(class_filter, str) or not isinstance(class_filter, Sequence):
        raise ValueError(
            f"{block}: class_filter must be a list of class names or null, got "
            f"{class_filter!r}"
        )
    names = tuple(class_filter)
    if not names:
        raise ValueError(f"{block}: {EMPTY_CLASS_FILTER_MESSAGE}")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{block}: class_filter holds class names, got {name!r}")

    return names


def class_filter_ids(
    names: Optional[Tuple[str, ...]],
    class_names: Sequence[str],
    *,
    block: str,
    request: ModelRequest,
) -> Optional[Tuple[int, ...]]:
    """The class ids that checked ``class_filter`` names keep for one model.

    Args:
        names: Names from ``checked_class_filter``, or ``None``.
        class_names: The model's class names.
        block: Block type, for error messages.
        request: The model's request, for error messages.

    Returns:
        Sorted class ids, or ``None`` to keep every class.

    Raises:
        ValueError: On a name the model does not have (with close matches).
    """
    if names is None:
        return None

    class_ids = _class_ids(names, class_names, block=block, request=request)

    return class_ids


def kept_class_ids(
    class_ids: Optional[Tuple[int, ...]], *, like: torch.Tensor
) -> Optional[torch.Tensor]:
    """``class_ids`` as a tensor on ``like``'s device; ``None`` keeps all.

    Build it once per call and pass it to ``keep_classes`` for every image.
    """
    if class_ids is None:
        return None

    kept = torch.tensor(class_ids, dtype=like.dtype, device=like.device)

    return kept


def keep_classes(
    prediction: Any, *, class_id: torch.Tensor, kept_ids: Optional[torch.Tensor]
) -> Any:
    """Keep the rows whose class is in ``kept_ids``; all rows when ``None``.

    Args:
        prediction: A native row-aligned prediction (``select_predictions``).
        class_id: The prediction's ``(N,)`` class ids.
        kept_ids: Accepted class ids on ``class_id``'s device, or ``None``.

    Returns:
        The selected prediction, or ``prediction`` itself without a filter.
    """
    if kept_ids is None:
        return prediction

    selected = select_predictions(prediction, mask=torch.isin(class_id, kept_ids))

    return selected


def prediction_metadata(
    model_metadata: Optional[Mapping[str, Any]],
    image: ImageData,
    *,
    class_names: Sequence[str],
    prediction_type: str,
) -> Dict[str, Any]:
    """What the model wrote, then the image's frame, class names and type."""
    metadata = {
        **(model_metadata or {}),
        **FrameRef.from_image(image).prediction_metadata(),
        "class_names": dict(enumerate(class_names)),
        "prediction_type": prediction_type,
    }

    return metadata


def identified_result(prediction: Any, *, model_id: str) -> Dict[str, Any]:
    """One image's outputs, with fresh row identity on its prediction.

    The row namespace is the image's ``inference_id``: one identity per
    independent result, minted after every metadata merge.

    Args:
        prediction: A row-bearing native prediction with its final metadata.
        model_id: The model id that served the image.

    Returns:
        ``{"predictions", "model_id", "inference_id"}``.
    """
    inference_id = uuid.uuid4().hex
    result = {
        "predictions": with_row_identity(prediction, namespace=inference_id),
        "model_id": model_id,
        "inference_id": inference_id,
    }

    return result


def require_rgb(image: ImageData, *, block: str) -> None:
    """Refuse an image that is not 3-channel RGB.

    Args:
        image: Image a model will read.
        block: Block type, for the error message.

    Raises:
        ValueError: When the image does not have 3 channels.
    """
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")


def _class_ids(
    class_filter: Tuple[str, ...],
    class_names: Sequence[str],
    *,
    block: str,
    request: ModelRequest,
) -> Tuple[int, ...]:
    ids_by_name = {name: class_id for class_id, name in enumerate(class_names)}
    unknown = [name for name in class_filter if name not in ids_by_name]
    if unknown:
        hints = {
            name: difflib.get_close_matches(name, list(class_names), n=1)
            for name in unknown
        }
        suggestions = ", ".join(
            f"{name!r} -> {matches[0]!r}" for name, matches in hints.items() if matches
        )
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise ValueError(
            f"{block}: class_filter names {unknown}, which model "
            f"{str(request)!r} does not have; its classes are "
            f"{list(class_names)}.{hint} Fix the names, or omit class_filter "
            "(or set it to null) to keep every class."
        )
    class_ids = tuple(sorted({ids_by_name[name] for name in class_filter}))

    return class_ids


def _ignored_keywords(post_process: Callable[..., Any]) -> frozenset:
    function = getattr(post_process, "__func__", post_process)
    name = f"{function.__module__}.{function.__qualname__}"
    ignored = IGNORED_KEYWORDS.get(name, frozenset())

    return ignored


def _unread_option_message(
    option: ModelOption,
    read: frozenset,
    options: Sequence[ModelOption],
    post_process: Callable[..., Any],
    *,
    block: str,
    request: ModelRequest,
) -> str:
    owner = type(getattr(post_process, "__self__", None)).__name__
    if option.keyword in _ignored_keywords(post_process):
        why = f"its post_process names '{option.keyword}' but never uses it"
    else:
        why = f"its post_process has no '{option.keyword}' parameter"
    supported = sorted(
        candidate.param for candidate in options if candidate.keyword in read
    )
    message = (
        f"{block}: model {str(request)!r} (backend {owner}) does not read "
        f"'{option.param}' ({why}), so {option.param}={option.value!r} would "
        f"have no effect. Options this model reads: {supported or 'none'}. Remove "
        f"'{option.param}' or set it to null for this model. Checked before any "
        "model runs."
    )

    return message


def _named_parameters(function: Callable[..., Any]) -> frozenset:
    parameters = inspect.signature(function).parameters.values()
    named = frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )

    return named


def _refuse_unsupplied(supplied: Mapping[Any, Any]) -> Callable[[ModelRequest], Any]:
    available = sorted(str(key) for key in supplied)

    def refuse(request: ModelRequest) -> Any:
        raise LookupError(
            f"model {str(request)!r} is not in the 'models' resource "
            f"(available: {available}); load it and pass it in that mapping when "
            "creating the session, or pass a ModelCache that can load it"
        )

    return refuse
