from transformers import AutoModelForImageTextToText
import torch
from inference.models.transformers import TransformerModel
from PIL import Image

class SmolVLM(TransformerModel):
    generation_includes_input = True
    transformers_class = AutoModelForImageTextToText
    load_base_from_roboflow = True
    is_chat_model = True
    model_id = "smolvlm2/smolvlm-2.2b-instruct"
    default_dtype = torch.bfloat16
    load_weights_as_transformers = True

    def __init__(self, *args, **kwargs):
        super().__init__(model_id=self.model_id, *args, **kwargs)

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_in},
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts