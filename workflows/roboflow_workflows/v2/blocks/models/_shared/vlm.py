"""Prompt and answer rules shared by the local VLM blocks.

Qwen backends read the system prompt from the text after a ``<system_prompt>``
marker in ``prompt``; the blocks append it there, as V1 does. A prompt or
system prompt that already contains the marker would be split in the wrong
place, so it fails the step.
"""

from typing import Any, Optional

from roboflow_workflows.v2.blocks.models.cache import ModelRequest

SYSTEM_PROMPT_MARKER = "<system_prompt>"

DEFAULT_PROMPT = "Describe what's in this image."


def prompt_with_system_prompt(
    prompt: Optional[str],
    system_prompt: Optional[str],
    *,
    default_system_prompt: str,
    family: str,
    block: str,
) -> str:
    """Build ``"<prompt><system_prompt><system prompt>"`` for one image.

    Args:
        prompt: The image's prompt; ``None`` or empty uses ``DEFAULT_PROMPT``.
        system_prompt: The call's system prompt; ``None`` or empty uses
            ``default_system_prompt``.
        default_system_prompt: The block's default system prompt.
        family: Model family name, for error messages.
        block: Block type, for error messages.

    Returns:
        The full prompt the backend splits at the marker.

    Raises:
        ValueError: When the prompt or system prompt contains the marker.
    """
    prompt = prompt or DEFAULT_PROMPT
    system_prompt = system_prompt or default_system_prompt
    for name, text in (("prompt", prompt), ("system_prompt", system_prompt)):
        if SYSTEM_PROMPT_MARKER in text:
            raise ValueError(
                f"{block}: {name} contains {SYSTEM_PROMPT_MARKER!r}, which "
                f"{family} reads as the start of the system prompt; "
                "remove it from the text"
            )
    full_prompt = f"{prompt}{SYSTEM_PROMPT_MARKER}{system_prompt}"

    return full_prompt


def single_text_answer(answers: Any, *, block: str, request: ModelRequest) -> str:
    """Return the one text answer a backend gave for one image.

    Args:
        answers: What the backend's ``prompt`` returned.
        block: Block type, for error messages.
        request: The model's request, for error messages.

    Returns:
        The answer text.

    Raises:
        ValueError: Unless ``answers`` is a list of exactly one text.
    """
    if (
        not isinstance(answers, list)
        or len(answers) != 1
        or not isinstance(answers[0], str)
    ):
        raise ValueError(
            f"{block}: model {str(request)!r} must answer one image with "
            f"a list of one text, got {answers!r}"
        )

    return answers[0]
