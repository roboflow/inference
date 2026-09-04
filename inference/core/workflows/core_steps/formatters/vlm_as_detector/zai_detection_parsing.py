# The Z.ai detection contracts, both xyxy lists normalized to 0-1000:
# GLM 5V Turbo (model_type "zai") uses the "box_2d" key and GLM 5.3 Flash
# (model_type "zai-flash") uses "bbox_2d". Both parse with the Qwen parser,
# which accepts either key; the model_types stay separate so saved
# workflows keep working if the contracts ever diverge again.
#
# `extract_zai_json_array` now lives in the shared VLM decoding package,
# where it is one of the format-agnostic JSON recovery strategies. It is
# re-exported here so the existing formatter blocks keep importing it from
# their historical path.

from inference.core.workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_zai_json_array,
)

__all__ = ["extract_zai_json_array"]
