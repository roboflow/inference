"""The four local VLM blocks (Qwen2.5-VL, Qwen3-VL, Qwen3.5 v2, SmolVLM2), in catalogue order."""

from roboflow_workflows.v2.blocks.models.qwen25vl.v1 import Qwen25VLBlock
from roboflow_workflows.v2.blocks.models.qwen3_5vl.v2 import Qwen35VLBlock
from roboflow_workflows.v2.blocks.models.qwen3vl.v1 import Qwen3VLBlock
from roboflow_workflows.v2.blocks.models.smolvlm2.v1 import SmolVLM2Block

BLOCKS = (Qwen25VLBlock, Qwen3VLBlock, Qwen35VLBlock, SmolVLM2Block)
