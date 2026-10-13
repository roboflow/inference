"""Steps every task-model block shares: models, request groups and options.

A task-model block runs one model call per phase and distinct request::

    network_input   check images and options, group rows by ModelRequest,
                    resolve every model, then pre_process each group
    raw_output      forward each group once
    result          post_process each group with the options its model reads,
                    then scatter the per-image results back to row order

Options reach a model only through its ``post_process`` keywords. An option
the step does not set gets its V1 default, and only when the model names that
keyword. An option the step sets for a model that does not read it fails the
step before any model runs: an NMS-free model must not appear to honour an IoU
threshold. A confidence mode (``"best"`` / ``"default"``) is not a set value:
a model without a ``confidence`` keyword applies its own threshold.

"Reads" means the keyword is named and used. Keywords collected by
``**kwargs`` do not count: no current task backend reads a block option from
``**kwargs`` or forwards it to a callee that does. ``IGNORED_KEYWORDS`` lists
the backends that name a keyword but never use it. Both facts come from a
source survey of the ``inference_models`` task backends; the task's
supported-keyword matrix records it.

``class_filter`` may differ per image. Rows that share a model request still
form one group and one forward; each row keeps its own classes.
"""

import difflib
import inspect
import uuid
from types import MappingProxyType
from typing import (
    Annotated,
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
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import (
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

MODELS_RESOURCE = "models"

# Literal ids, or selectors; ids that vary per image arrive as a Batch. A
# selected package may be None (the loader chooses); a selected model id may not.
ModelIdField = StrictStr | Ref(STRING_KIND, batch="if_varying")
PackageIdField = Optional[StrictStr] | Ref(
    STRING_KIND, batch="if_varying", allow_none=True
)
# A literal list, or a selector; a filter that varies per image arrives as a
# Batch whose members may be None (keep every class of that image).
ClassFilterParam = Optional[List[StrictStr]] | Ref(
    LIST_OF_VALUES_KIND, batch="if_varying", allow_none=True
)
RequestField = Union[None, str, Batch]
ClassFilterField = Union[None, Sequence[str], Batch]

MODEL_ID_DESCRIPTION = (
    "Model id (or local path) of the model in the 'models' resource; a "
    "selector may choose it per run or per image. An image whose per-image "
    "model id is None or filtered out is skipped and gets no prediction."
)
PACKAGE_ID_DESCRIPTION = (
    "Exact model package to load; None lets the loader choose. A selector may "
    "choose it per run or per image, and a selected None also lets the loader "
    "choose for that image. An image whose selected package is filtered out is "
    "skipped. A model supplied without a package id rejects a package. List a "
    "model's package ids with "
    'inference_models.AutoModel.describe_model("<model_id>").'
)

CONFIDENCE_MODES_DESCRIPTION = (
    "'best' uses the model's evaluated per-class thresholds (then its global "
    "one, then its default); 'default' uses the model's default threshold; a "
    "number 0..1 applies to every class."
)

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

UnitFloat = Annotated[StrictFloat, Field(ge=0, le=1)]

MODEL_ID_OUTPUT = Output(
    STRING_KIND, source="image", description="Model id that served this image."
)
INFERENCE_ID_OUTPUT = Output(
    STRING_KIND,
    source="image",
    description=(
        "Fresh id of this image's result; for results with rows, also the "
        "namespace of their row identity."
    ),
)


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


class ModelGroup(NamedTuple):
    """Rows of one call that share a request, preprocessed by its model.

    ``options`` are the ``post_process`` keywords bound for this model;
    ``class_ids`` holds, per row of ``rows``, the ids ``class_filter`` keeps,
    ``None`` for all classes.
    """

    request: ModelRequest
    model: Any
    rows: Tuple[int, ...]
    batch: Any
    metadata: Any
    options: Mapping[str, Any]
    class_ids: Tuple[Optional[Tuple[int, ...]], ...]


class ModelSelectionParams(BlockParams):
    """``model_id`` and ``model_package_id`` of every task-model block."""

    model_id: ModelIdField = Field(
        description=MODEL_ID_DESCRIPTION,
        examples=["rfdetr-nano", "yolov8n-640", "$inputs.model_id"],
    )
    model_package_id: PackageIdField = Field(
        default=None,
        description=PACKAGE_ID_DESCRIPTION,
        examples=[None, "$inputs.model_package_id"],
    )

    @field_validator("model_id", "model_package_id")
    @classmethod
    def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
        if value == "":
            raise ValueError("must be a non-empty string")

        return value


class NmsParams(ModelSelectionParams):
    """Class filter and NMS options of the box-producing model blocks."""

    class_filter: ClassFilterParam = Field(
        default=None,
        description=(
            "Class names to keep; omit or null to keep every class. Every name "
            "must be a class of the serving model. A selector may choose it per "
            "run or per image; a per-image null keeps every class of that image. "
            "Applied after the model's own post-processing; images that share a "
            "model still run in one model call."
        ),
        examples=[["person", "car"], "$inputs.class_filter"],
    )
    iou_threshold: Optional[StrictFloat] | Ref(FLOAT_KIND) = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            f"NMS IoU threshold, 0..1. Unset: {DEFAULT_IOU_THRESHOLD} for models "
            "with NMS. Setting it for a model without NMS (e.g. RF-DETR) fails "
            "the step."
        ),
        examples=[0.5, "$inputs.iou_threshold"],
    )
    max_detections: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
        default=None,
        ge=1,
        description=(
            f"Maximum detections per image. Unset: {DEFAULT_MAX_DETECTIONS} for "
            "models that read it. Setting it for a model that does not read it "
            "fails the step."
        ),
        examples=[100, "$inputs.max_detections"],
    )
    class_agnostic_nms: Optional[StrictBool] | Ref(BOOLEAN_KIND) = Field(
        default=None,
        description=(
            "Run NMS across classes. Unset: False for models with NMS. Setting "
            "it for a model without NMS fails the step."
        ),
        examples=[True, "$inputs.class_agnostic_nms"],
    )

    @field_validator("class_filter")
    @classmethod
    def _reject_empty_filter(cls, value: Any) -> Any:
        if isinstance(value, list) and not value:
            raise ValueError(EMPTY_CLASS_FILTER_MESSAGE)

        return value


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
            f"{block} needs the '{MODELS_RESOURCE}' resource as a mapping of model "
            f"id to loaded model, or a ModelCache, got {type(models).__name__}"
        )

    # The snapshot keeps the model objects; nothing is loaded.
    supplied = MappingProxyType(dict(models))
    cache = ModelCache(_refuse_unsupplied(supplied), supplied=supplied)

    return cache


def preprocess_groups(
    image: Batch,
    *,
    cache: ModelCache,
    model_id: RequestField,
    model_package_id: RequestField,
    options: Sequence[ModelOption],
    block: str,
    class_filter: ClassFilterField = None,
    returns_metadata: bool = True,
) -> Tuple[ModelGroup, ...]:
    """Group rows by request, resolve all models, then preprocess each group.

    Every model is resolved and every option and class name checked before
    the first model call, so one bad row request or option runs nothing.

    Args:
        image: The batch of RGB images of one call.
        cache: Source of the models.
        model_id: Model id for every image, or a ``Batch`` with one per image.
        model_package_id: Package id for every image, a ``Batch`` with one
            per image, or ``None``.
        options: Step options to bind to each model's ``post_process``.
        block: Block type, for error messages.
        class_filter: Class names to keep for every image, a ``Batch`` with
            one list (or ``None``) per image, or ``None``; ``None`` keeps
            every class. Differing filters do not split a group.
        returns_metadata: Whether ``pre_process`` returns ``(batch,
            metadata)``; classifiers return the batch alone.

    Returns:
        One group per distinct request, in order of first appearance.

    Raises:
        ValueError: On a non-RGB image, a request or filter batch of the wrong
            length, an option the model does not read, an empty class filter
            or one naming a class the model does not have.
    """
    for member in image:
        _require_rgb(member, block=block)
    row_filters = tuple(
        _checked_class_filter(row_filter, block=block)
        for row_filter in _per_row(
            class_filter, row_count=len(image), name="class_filter"
        )
    )
    rows_by_request = _rows_by_request(
        len(image), model_id=model_id, model_package_id=model_package_id
    )
    models = {request: cache.get(request) for request in rows_by_request}
    bindings = {
        request: (
            post_process_options(
                model.post_process, options, block=block, request=request
            ),
            _row_class_ids(
                [row_filters[row] for row in rows_by_request[request]],
                model.class_names,
                block=block,
                request=request,
            ),
        )
        for request, model in models.items()
    }

    members = image.content
    groups = []
    for request, rows in rows_by_request.items():
        model = models[request]
        processed = model.pre_process(
            [members[row].tensor_image for row in rows], input_color_format="rgb"
        )
        batch, metadata = processed if returns_metadata else (processed, None)
        group_options, class_ids = bindings[request]
        groups.append(
            ModelGroup(request, model, rows, batch, metadata, group_options, class_ids)
        )

    return tuple(groups)


def forward_groups(groups: Tuple[ModelGroup, ...]) -> Tuple[Any, ...]:
    """Run each group's network once on its rows (``forward``)."""
    raw_output = tuple(group.model.forward(group.batch) for group in groups)

    return raw_output


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


def checked_count(predictions: Any, group: ModelGroup, *, block: str) -> None:
    """Fail when a model returns another number of results than images."""
    if len(predictions) != len(group.rows):
        raise ValueError(
            f"{block}: post_process of model {str(group.request)!r} returned "
            f"{len(predictions)} predictions for {len(group.rows)} images"
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
        ValueError: On another string or a number outside 0..1.
    """
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


def class_filter_tensors(
    group: ModelGroup, *, like: torch.Tensor
) -> Tuple[Optional[torch.Tensor], ...]:
    """Each row's kept class ids on ``like``'s device; ``None`` keeps all.

    One tensor per distinct filter of the group, shared by its rows; pass a
    row's entry to ``keep_classes``.
    """
    tensors: Dict[Tuple[int, ...], torch.Tensor] = {}
    for class_ids in group.class_ids:
        if class_ids is not None and class_ids not in tensors:
            tensors[class_ids] = torch.tensor(
                class_ids, dtype=like.dtype, device=like.device
            )
    per_row = tuple(
        None if class_ids is None else tensors[class_ids]
        for class_ids in group.class_ids
    )

    return per_row


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


def model_dependency(params: Any, *, task_type: str) -> List[DependentResource]:
    """The one ``model`` resource a task-model step needs from ``models``.

    Args:
        params: Validated step parameters with ``model_id`` and
            ``model_package_id``.
        task_type: The block's prediction type.

    Returns:
        One resource named by ``model_id``, a selector string when it is
        chosen at run time, with its ``model_package_id``.
    """
    resources = [
        DependentResource(
            resource_type="model",
            identifier=params.model_id,
            details={
                "resource": MODELS_RESOURCE,
                "task_type": task_type,
                "model_package_id": params.model_package_id,
            },
        )
    ]

    return resources


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


def _rows_by_request(
    row_count: int, *, model_id: RequestField, model_package_id: RequestField
) -> Dict[ModelRequest, Tuple[int, ...]]:
    """Group row indices by request, in order of first appearance."""
    model_ids = _per_row(model_id, row_count=row_count, name="model_id")
    package_ids = _per_row(
        model_package_id, row_count=row_count, name="model_package_id"
    )
    rows_by_request: Dict[ModelRequest, List[int]] = {}
    for row, (row_model_id, row_package_id) in enumerate(zip(model_ids, package_ids)):
        request = ModelRequest(row_model_id, row_package_id)
        rows_by_request.setdefault(request, []).append(row)
    grouped = {request: tuple(rows) for request, rows in rows_by_request.items()}

    return grouped


def _checked_class_filter(
    class_filter: Optional[Sequence[str]], *, block: str
) -> Optional[Tuple[str, ...]]:
    # A selected filter skipped the literal validators; check it the same way.
    if class_filter is None:
        return None

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


def _row_class_ids(
    row_filters: Sequence[Optional[Tuple[str, ...]]],
    class_names: Sequence[str],
    *,
    block: str,
    request: ModelRequest,
) -> Tuple[Optional[Tuple[int, ...]], ...]:
    # Each distinct filter is checked and resolved once per model.
    resolved: Dict[Tuple[str, ...], Tuple[int, ...]] = {}
    for row_filter in row_filters:
        if row_filter is not None and row_filter not in resolved:
            resolved[row_filter] = _class_ids(
                row_filter, class_names, block=block, request=request
            )
    class_ids = tuple(
        None if row_filter is None else resolved[row_filter]
        for row_filter in row_filters
    )

    return class_ids


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


def _per_row(value: RequestField, *, row_count: int, name: str) -> Tuple[Any, ...]:
    if not isinstance(value, Batch):
        return (value,) * row_count

    if len(value) != row_count:
        raise ValueError(f"{name} has {len(value)} values for {row_count} images")

    return value.content


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
            f"model {str(request)!r} is not in the '{MODELS_RESOURCE}' resource "
            f"(available: {available}); load it and pass it in that mapping when "
            "creating the session, or pass a ModelCache that can load it"
        )

    return refuse


def _require_rgb(image: ImageData, *, block: str) -> None:
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")
