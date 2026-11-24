"""Row checks shared by the local VLM, Moondream2, Cosmos, Florence-2 and GLM-OCR blocks.

A per-image value (such as a prompt) reaches ``run()`` as one value for every
image, or as a ``Batch`` with one value per image. Such a ``Batch`` must carry
the image batch's logical indices in the same order: equal length alone does
not show that its rows describe the same images, so a mismatch fails before
any model is loaded or invoked.

Per-image checks name the failing image by its logical index, as the engine's
own per-row validation does.
"""

import inspect
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from roboflow_workflows.v2.blocks.models.cache import ModelRequest
from roboflow_workflows.v2.data.batch import Batch, Index


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


@contextmanager
def image_row(index: Index, *, block: str) -> Iterator[None]:
    """Name the block and one image's logical index in a ValueError raised inside.

    Args:
        index: Logical index of the image whose values are checked.
        block: Block type, for error messages.

    Raises:
        ValueError: The raised error, prefixed with
            ``"<block>: image at index [i]: "``.
    """
    try:
        yield
    except ValueError as error:
        raise ValueError(f"{block}: image at index {list(index)}: {error}") from error


def require_rgb_images(image: Batch, *, block: str) -> None:
    """Fail unless every image has three (RGB) channels, naming the first that does not.

    Args:
        image: The step's image batch.
        block: Block type, for the indexed channel-count diagnostic.

    Raises:
        ValueError: On an image with another channel count.
    """
    for index, member in zip(image.indices, image):
        if member.channels == 3:
            continue

        with image_row(index, block=block):
            raise ValueError(
                "a local vision model needs a 3-channel RGB image, got "
                f"{member.channels} channel(s)"
            )


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


def check_backend_call(
    model: Any,
    method: str,
    *,
    block: str,
    request: ModelRequest,
    options: Dict[str, int],
    **keywords: Any,
) -> None:
    """Fail before any model runs when a backend method cannot take the call.

    The block calls ``model.<method>([tensor], **keywords, **options)`` once
    per image; this binds that call without running it.

    Args:
        model: The resolved model.
        method: Name of the backend method the block calls.
        block: Block type, for error messages.
        request: The model's request, for error messages.
        options: Generation keywords from ``generation_options``.
        **keywords: The other keywords of every call.

    Raises:
        ValueError: When the method is missing or rejects the call. The
            message names the model, its backend class and the fix.
    """
    where = f"{block}: model {str(request)!r} (backend {type(model).__name__})"
    function = getattr(model, method, None)
    if not callable(function):
        raise ValueError(f"{where} has no {method}() method")

    signature = inspect.signature(function)
    if _binds(signature, {**keywords, **options}):
        return

    if options and _binds(signature, keywords):
        fix = f"Remove {', '.join(options)} or set it to null for this model"
    else:
        fix = f"Supply a model whose {method}() accepts this call"
    called = ", ".join(["images", *keywords, *options])
    raise ValueError(
        f"{where} does not accept {method}({called}). {fix}. Checked before any "
        "model runs."
    )


def _binds(signature: inspect.Signature, keywords: Dict[str, Any]) -> bool:
    try:
        signature.bind([], **keywords)
    except TypeError:
        return False

    return True
