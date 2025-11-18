import torch
import time

from sam3.model.sam3_image_processor import Sam3Processor
from sam3 import build_sam3_image_model

from PIL import Image
from tqdm import tqdm

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


sam3_root = "/home/hansent/inference-private/sam3_private"
bpe_path = "/tmp/cache/sam3/sam3_final/bpe_simple_vocab_16e6.txt.gz"
checkpoint_path = "/tmp/cache/sam3/sam3_final/weights.pt"
has_presence_token = True


model = build_sam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path=checkpoint_path,
    has_presence_token=has_presence_token,
)
processor = Sam3Processor()

image_path = f"{sam3_root}/assets/images/test_image.jpg"
image = Image.open(image_path)


def run_full_inference():
    inference_state = processor(image, instance_prompt=False)
    processor.reset_state(inference_state)
    processor.add_prompt(inference_state, text_str="shoe", instance_prompt=False)
    model.run_inference(inference_state)
    out = processor.postprocess_output(inference_state, output_prob_thresh=0.5)
    return out


# Prepare inference state once, (warmup if torch compiled model)
t0_init = time.perf_counter()
run_full_inference()
first_init_elapsed = time.perf_counter() - t0_init
print(f"First inference time: {first_init_elapsed:.4f}s")

tqdm.write("Processing image with text prompt")
for _ in tqdm(range(1000)):
    run_full_inference()
