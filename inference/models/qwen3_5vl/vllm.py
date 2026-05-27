import base64
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import requests

from inference.core.utils.image_utils import encode_image_to_jpeg_bytes

DEFAULT_QWEN_SYSTEM_PROMPT = "You are a helpful assistant."


class QwenVLLMInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenPromptParts:
    prompt: str
    system_prompt: str


class QwenVLLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        request_timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._request_timeout = request_timeout

    def generate(
        self,
        model_id: str,
        image_bgr: np.ndarray,
        prompt: Optional[str],
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
    ) -> Union[str, Dict[str, str]]:
        prompt_parts = split_qwen_prompt(prompt=prompt)
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": prompt_parts.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_bgr_to_data_url(image_bgr=image_bgr)
                            },
                        },
                        {"type": "text", "text": prompt_parts.prompt},
                    ],
                },
            ],
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            response_payload = response.json()
        except Exception as error:
            raise QwenVLLMInferenceError(
                f"vLLM chat completion request failed for model {model_id}: {error}"
            ) from error
        return parse_chat_completion_output(
            response_payload=response_payload,
            enable_thinking=enable_thinking,
        )


def split_qwen_prompt(prompt: Optional[str]) -> QwenPromptParts:
    if prompt is None:
        return QwenPromptParts(
            prompt="Describe what's in this image.",
            system_prompt=DEFAULT_QWEN_SYSTEM_PROMPT,
        )
    split_prompt = prompt.split("<system_prompt>")
    if len(split_prompt) == 1:
        return QwenPromptParts(
            prompt=split_prompt[0] or "Describe what's in this image.",
            system_prompt=DEFAULT_QWEN_SYSTEM_PROMPT,
        )
    return QwenPromptParts(
        prompt=split_prompt[0] or "Describe what's in this image.",
        system_prompt=split_prompt[1] or DEFAULT_QWEN_SYSTEM_PROMPT,
    )


def image_bgr_to_data_url(image_bgr: np.ndarray) -> str:
    jpeg_bytes = encode_image_to_jpeg_bytes(image=image_bgr)
    base64_image = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{base64_image}"


def parse_chat_completion_output(
    response_payload: dict,
    enable_thinking: bool,
) -> Union[str, Dict[str, str]]:
    try:
        message = response_payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise QwenVLLMInferenceError(
            "vLLM chat completion response did not include choices[0].message"
        ) from error
    answer = message.get("content") or ""
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("reasoning_text")
    )
    if enable_thinking:
        if reasoning is not None:
            return {"thinking": str(reasoning).strip(), "answer": answer.strip()}
        return split_thinking_from_answer(text=answer)
    return strip_thinking(text=answer)


def split_thinking_from_answer(text: str) -> Dict[str, str]:
    text = ensure_open_think_tag(text=text)
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        return {"thinking": thinking, "answer": answer}
    return {"thinking": text.replace("<think>", "").strip(), "answer": ""}


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def ensure_open_think_tag(text: str) -> str:
    if text.lstrip().startswith("<think>"):
        return text
    if "</think>" in text:
        return "<think>" + text
    return text
