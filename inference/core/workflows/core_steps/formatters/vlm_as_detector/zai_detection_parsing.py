# The Z.ai detection contracts, both "box_2d" lists normalized to 0-1000:
# GLM 5V Turbo (model_type "zai") uses xyxy order and parses with the Qwen
# parser; GLM 5.3 Flash (model_type "zai-flash") uses yxyx order and parses
# with the Gemini parser. The axis order is not recoverable from the data,
# so each model maps to its own model_type in the registry.
#
# `extract_zai_json_array` now lives in the shared VLM decoding package,
# where it is one of the format-agnostic JSON recovery strategies. It is
# re-exported here so the existing formatter blocks keep importing it from
# their historical path.

from inference.core.workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_zai_json_array,
)

__all__ = ["extract_zai_json_array"]
