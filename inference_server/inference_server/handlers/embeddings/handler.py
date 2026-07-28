from __future__ import annotations

import asyncio
import math

from inference_server.framework.entities import ServerHooks
from inference_server.proxies.base import ModelManagerProxy


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _flatten(embedding) -> list[float]:
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    while isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
        embedding = embedding[0]
    return embedding


async def _embed_image(proxy, common, hooks, image: bytes):
    return await proxy.infer(
        model_id=common.model_id,
        image=image,
        task="embed_images",
        instance=common.instance,
        params={},
        request=hooks.request,
    )


async def _embed_texts(proxy, common, hooks, texts: list[str]):
    return await proxy.infer(
        model_id=common.model_id,
        image=None,
        task="embed_text",
        instance=common.instance,
        params={"texts": texts},
        request=hooks.request,
    )


async def handle_embeddings(
    action: str,
    input_data: dict,
    proxy: ModelManagerProxy,
    hooks: ServerHooks,
):
    images: list[bytes] = input_data["images"]
    params: dict = input_data["params"]
    common = hooks.common

    if action == "embed_images":
        if len(images) == 1:
            return await proxy.infer(
                model_id=common.model_id,
                image=images[0],
                task="embed_images",
                instance=common.instance,
                params=params,
                request=hooks.request,
            )
        return await asyncio.gather(
            *(_embed_image(proxy, common, hooks, img) for img in images)
        )

    if action == "embed_text":
        return await _embed_texts(proxy, common, hooks, params["texts"])

    subject_text = params.get("subject_text")
    prompt_texts = params.get("prompt_texts") or []
    if isinstance(prompt_texts, str):
        prompt_texts = [prompt_texts]

    if subject_text:
        subject_emb = _flatten(await _embed_texts(proxy, common, hooks, [subject_text]))
        prompt_images = images
    else:
        subject_emb = _flatten(await _embed_image(proxy, common, hooks, images[0]))
        prompt_images = images[1:]

    prompt_embs = []
    if prompt_images:
        prompt_embs.extend(
            _flatten(e)
            for e in await asyncio.gather(
                *(_embed_image(proxy, common, hooks, img) for img in prompt_images)
            )
        )
    if prompt_texts:
        text_embs = await _embed_texts(proxy, common, hooks, prompt_texts)
        if hasattr(text_embs, "tolist"):
            text_embs = text_embs.tolist()
        prompt_embs.extend(_flatten([e]) for e in text_embs)

    return {
        "type": "roboflow-comparison-v1",
        "similarity": [_cosine(subject_emb, p) for p in prompt_embs],
    }
