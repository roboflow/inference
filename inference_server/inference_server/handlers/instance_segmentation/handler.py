from __future__ import annotations

from typing import Any

from inference_server.framework.entities import ServerHooks
from inference_server.framework.fanout import gather_bounded


async def handle_instance_segmentation(
    action: str,
    input_data: dict,
    proxy: Any,
    hooks: ServerHooks,
):
    images: list[bytes] = input_data["images"]
    params: dict = input_data["params"]
    common = hooks.common
    task = action if action and action != "infer" else None

    if len(images) == 1:
        return await proxy.infer(
            model_id=common.model_id,
            image=images[0],
            task=task,
            instance=common.instance,
            params=params,
            request=hooks.request,
        )

    return await gather_bounded(
        *(
            proxy.infer(
                model_id=common.model_id,
                image=img,
                task=task,
                instance=common.instance,
                params=params,
                request=hooks.request,
            )
            for img in images
        )
    )
