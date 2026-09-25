from __future__ import annotations

from typing import Any

from inference_server.framework.entities import ServerHooks
from inference_server.framework.fanout import gather_bounded


async def handle_interactive_instance_segmentation(
    action: str,
    input_data: dict,
    proxy: Any,
    hooks: ServerHooks,
):
    images: list[bytes] = input_data["images"]
    params: dict = input_data["params"]
    common = hooks.common
    dispatch_action = action if action and action != "infer" else None

    if not images:
        return await proxy.infer(
            model_id=common.model_id,
            image=b"",
            action=dispatch_action,
            instance=common.instance,
            params=params,
            request=hooks.request,
        )

    if len(images) == 1:
        return await proxy.infer(
            model_id=common.model_id,
            image=images[0],
            action=dispatch_action,
            instance=common.instance,
            params=params,
            request=hooks.request,
        )

    hashes = params.get("image_hashes")
    if hashes is None:
        per_call_params = [params] * len(images)
    elif isinstance(hashes, list) and len(hashes) == len(images):
        per_call_params = [{**params, "image_hashes": [h]} for h in hashes]
    else:
        raise ValueError("image_hashes count must match the number of images")

    return await gather_bounded(
        *(
            proxy.infer(
                model_id=common.model_id,
                image=img,
                action=dispatch_action,
                instance=common.instance,
                params=call_params,
                request=hooks.request,
            )
            for img, call_params in zip(images, per_call_params)
        )
    )
