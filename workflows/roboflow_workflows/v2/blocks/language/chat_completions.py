"""Message and usage helpers shared by chat-completions providers."""

from typing import Any, Dict, List, Optional, Tuple

from roboflow_workflows.v2.blocks.language.chat import ChatPrompt, token_count


def chat_messages(
    chat_prompt: ChatPrompt, image_urls: List[str]
) -> List[Dict[str, Any]]:
    """Format a prompt as chat-completions ``messages``.

    Args:
        chat_prompt: System text and user text; empty texts are left out.
        image_urls: ``data:`` or http URLs, sent after the text.

    Returns:
        An optional system message and one user message.
    """
    messages: List[Dict[str, Any]] = []
    if chat_prompt.system:
        messages.append({"role": "system", "content": chat_prompt.system})

    content: List[Dict[str, Any]] = []
    if chat_prompt.text:
        content.append({"type": "text", "text": chat_prompt.text})
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    messages.append({"role": "user", "content": content})

    return messages


def chat_completion_usage(usage: Any) -> Tuple[Optional[int], Optional[int]]:
    """Read ``prompt_tokens`` / ``completion_tokens`` from a usage object or dict.

    Args:
        usage: SDK usage object, a dict or ``None``.

    Returns:
        ``(input_tokens, output_tokens)``; ``None`` for a missing count.
    """
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        counts = (
            token_count(usage.get("prompt_tokens")),
            token_count(usage.get("completion_tokens")),
        )
        return counts

    counts = (
        token_count(getattr(usage, "prompt_tokens", None)),
        token_count(getattr(usage, "completion_tokens", None)),
    )

    return counts
