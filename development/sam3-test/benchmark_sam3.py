from sam3 import build_sam3_image_model
import torch

from tqdm import tqdm

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

torch.set_grad_enabled(False)

checkpoint_path = "/tmp/cache/sam3/sam3_image_model_only/weights.pt"
bpe_path = "/tmp/cache/sam3/sam3_image_model_only/bpe_simple_vocab_16e6.txt.gz"


model = build_sam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path=checkpoint_path,
    device="cuda",
    eval_mode=True,
)


image_path = "/home/hansent/images/traffic.jpg"


def run_once():
    inference_state = model.init_state(image_path)
    model.reset_state(inference_state)
    model.add_prompt(
        inference_state,
        text_str="floor",
        output_prob_thresh=0.5,
        instance_prompt=False,
    )


for _ in tqdm(range(1000)):
    run_once()
