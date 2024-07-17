from transformers import PaliGemmaForConditionalGeneration

from inference.models.transformers import LoRATransformerModel, TransformerModel


class PaliGemma(TransformerModel):
    """By using you agree to the terms listed at https://ai.google.dev/gemma/terms"""

    generation_includes_input = True
    transformers_class = PaliGemmaForConditionalGeneration


class LoRAPaliGemma(LoRATransformerModel):
    """By using you agree to the terms listed at https://ai.google.dev/gemma/terms"""

    generation_includes_input = True
    transformers_class = PaliGemmaForConditionalGeneration
    load_base_from_roboflow = True
