from transformers import AutoConfig, AutoTokenizer
import accelerate
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from huggingface_hub import notebook_login
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN

import torch
model_path = "SkunkworksAI/BakLLaVA-1"

cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
low_gpu_memory_optimization = True
if low_gpu_memory_optimization:
    model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, load_in_8bit=True, config=cfg_pretrained, cache_dir="/cache")
else:
    model = LlavaMistralForCausalLM.from_pretrained(model_path, config=cfg_pretrained, cache_dir="/cache")
    model.to("cuda")

mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
if mm_use_im_patch_token:
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
if mm_use_im_start_end:
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.float16)
image_processor = vision_tower.image_processor

if hasattr(model.config, "max_sequence_length"):
    context_len = model.config.max_sequence_length
else:
    context_len = 2048

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


# image = load_image("https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg")
image = load_image("https://cdn.discordapp.com/attachments/1096822099345145969/1164641565550067852/heart_1.png?ex=6543f3fb&is=65317efb&hm=448cb26e19c141871e776af98077c4c1e97a8f29b96916ab671e5010c00e3625&")
image = load_image("https://www.encyclopedie-environnement.org/app/uploads/2016/06/symbiose-evolution_focus.jpg")
if low_gpu_memory_optimization:
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
else:
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()

query = "Describe this image"

if model.config.mm_use_im_start_end:
    query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
else:
    query = DEFAULT_IMAGE_TOKEN + '\n' + query

conv = conv_templates["llava_v1"].copy()

conv.append_message(conv.roles[0], query)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids=input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=1024,
        use_cache=True,
        stopping_criteria=[stopping_criteria])

input_token_len = input_ids.shape[1]
n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
if n_diff_input_output > 0:
    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)]
outputs = outputs.strip()
print(outputs)
