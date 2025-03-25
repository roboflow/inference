from transformers import AutoProcessor, AutoModelForImageTextToText

from inference.models.transformers import TransformerModel


class SmolVLM(TransformerModel):
    generation_includes_input = True
    transformers_class = AutoModelForImageTextToText
    load_base_from_roboflow = False
    is_chat_model = True