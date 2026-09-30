"""Startup loading of the OWLv2 models listed in ``PRELOAD_HF_IDS``."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_OWLV2_REGISTRY_PREFIX = "owlv2/"


def hf_id_to_model_id(hf_id: str) -> str:
    """Map a Hugging Face OWLv2 id onto its ``owlv2/<name>`` registry id.

    Args:
        hf_id: Hugging Face id such as ``google/owlv2-base-patch16-ensemble``,
            or a bare model name.

    Returns:
        Registry id built from the part after the last ``/``.
    """
    name = hf_id.rsplit("/", 1)[-1]
    model_id = f"{_OWLV2_REGISTRY_PREFIX}{name}"

    return model_id


async def preload_hf_models(
    gateway: Any, hf_ids: list[str], *, api_key: Optional[str]
) -> None:
    """Load each ``PRELOAD_HF_IDS`` entry through the gateway, unpinned.

    Failures are logged and swallowed; readiness never waits on these loads.

    Args:
        gateway: Gateway exposing ``load(model_id, api_key, timeout_s, pinned)``.
        hf_ids: Hugging Face OWLv2 ids to load, one after another.
        api_key: Roboflow API key used to fetch the weights.
    """
    for hf_id in hf_ids:
        model_id = hf_id_to_model_id(hf_id)
        try:
            result = await gateway.load(
                model_id, api_key=api_key, timeout_s=300.0, pinned=False
            )
        except Exception:
            logger.error("Preload of HF model '%s' failed", hf_id, exc_info=True)
            continue

        if result and result[0] == "ok":
            logger.info("Preload of HF model '%s' as '%s': %s", hf_id, model_id, result)
        else:
            logger.error("Preload of HF model '%s' failed: %s", hf_id, result)
