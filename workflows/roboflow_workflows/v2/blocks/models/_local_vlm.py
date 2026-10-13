"""Row and request helpers of the local Moondream2, Cosmos and Florence-2 blocks.

A selectable value reaches ``run()`` as one value for every image, or as a
``Batch`` with one value per image. Such a ``Batch`` must carry the image
batch's logical indices in the same order. Equal length alone does not show
that its rows describe the same images, so a mismatch fails before any model
is loaded or invoked. Compiled workflows deliver every per-image column with
the image batch's indices.
"""

from typing import Any, Dict, Optional, Tuple, Union

from pydantic import StrictInt, StrictStr
from roboflow_workflows.v2.blocks.models.cache import ModelRequest
from roboflow_workflows.v2.blocks.models.common import RequestField
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Ref

# Params annotations: a literal, or a selector. Text may vary per image; the
# generation limit is one value for the whole step.
TextParam = Optional[StrictStr] | Ref(STRING_KIND, batch="if_varying", allow_none=True)
TokensParam = Optional[StrictInt] | Ref(INTEGER_KIND, allow_none=True)
# What run() receives for them.
TextField = Union[None, str, Batch]
TokensField = Optional[int]


def per_row(value: Any, *, image: Batch, name: str) -> Tuple[Any, ...]:
    """Return one value per image, in image order.

    Args:
        value: One value for every image, or a ``Batch`` with one per image.
        image: The step's image batch.
        name: Parameter name, for error messages.

    Returns:
        ``value`` repeated for every image, or the ``Batch`` content.

    Raises:
        ValueError: When a ``Batch`` has another length than ``image``, or
            other logical indices, including the same indices in another order.
    """
    if not isinstance(value, Batch):
        return (value,) * len(image)

    if len(value) != len(image):
        raise ValueError(f"{name} has {len(value)} values for {len(image)} images")

    if value.indices != image.indices:
        raise ValueError(
            f"{name} rows {list(value.indices)} do not match image rows "
            f"{list(image.indices)}; a per-image Batch needs the image indices "
            "in the same order"
        )

    return value.content


def require_rgb(image: Batch) -> None:
    """Fail unless every image has three (RGB) channels.

    Args:
        image: The step's image batch.

    Raises:
        ValueError: On an image with another channel count.
    """
    for member in image:
        if member.channels != 3:
            raise ValueError("A local vision model needs a 3-channel RGB image")


def model_requests(
    image: Batch, *, model_id: RequestField, model_package_id: RequestField
) -> Tuple[ModelRequest, ...]:
    """Return each image's exact model request.

    Args:
        image: The step's image batch.
        model_id: Model id for every image, or a ``Batch`` with one per image.
        model_package_id: Package id for every image, a ``Batch`` with one per
            image, or ``None``.

    Returns:
        One ``ModelRequest`` per image, in image order.

    Raises:
        ValueError: On an empty or non-text id, or a ``Batch`` not aligned
            with ``image``.
    """
    model_ids = per_row(model_id, image=image, name="model_id")
    package_ids = per_row(model_package_id, image=image, name="model_package_id")
    requests = tuple(map(ModelRequest, model_ids, package_ids))

    return requests


def checked_text(value: Any, *, name: str) -> Optional[str]:
    """Return a resolved text value after checking that it is text or ``None``.

    Args:
        value: One image's resolved value.
        name: Parameter name, for error messages.

    Returns:
        ``value`` unchanged.

    Raises:
        ValueError: On a value that is neither text nor ``None``.
    """
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be text or None, got {value!r}")

    return value


def generation_options(max_new_tokens: Any) -> Dict[str, int]:
    """Return the generation keywords of one backend call.

    Args:
        max_new_tokens: Positive limit, or ``None`` for the backend default.

    Returns:
        ``{}`` for ``None``, so the backend keyword default applies, otherwise
        ``{"max_new_tokens": max_new_tokens}``.

    Raises:
        ValueError: On a value other than ``None`` or a positive integer.
    """
    if max_new_tokens is None:
        return {}

    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer or None")

    options = {"max_new_tokens": max_new_tokens}

    return options
