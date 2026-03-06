import io
import base64
import time
import torch
import re
import os
import yaml
from PIL import Image
import random
import json
import torchvision.transforms.functional as TF
from openai import OpenAI, AzureOpenAI
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoringExample:
    input_full_highlighted: torch.tensor
    input_cropped_highlighted: torch.tensor
    input_cropped: torch.tensor
    rendered: torch.tensor
    rationale: Optional[str] = None


class Client:
    def __init__(self, model_config_path, model_name):
        self.model_configs = load_yaml(model_config_path)
        config = self.model_configs.get(model_name)
        if config is None:
            raise ValueError(f"Unknown model_name: {model_name}")
        # os.environ["OPENAI_API_KEY"] = config["api_key"]
        if config["provider"] == "openai":
            client = OpenAI()
        elif config["provider"] == "azure":
            client = AzureOpenAI(
                api_version=config["api_version"],
                api_key=config["api_key"],
                azure_endpoint=config["azure_endpoint"],
            )
        else:
            raise ValueError(f"Unknown provider: {config['provider']}")
        self.client = client
        self.deployed_name = config["model_name"]

    def query(self, content, temperature=1e-5, top_p=1):
        completion = self.client.chat.completions.create(
            model=self.deployed_name,
            messages=content,
            temperature=temperature,
            top_p=top_p,
        )
        response = completion.choices[0].message.content
        return response


class GPTScorer:
    def __init__(
        self, model_name, prompt_path, icl_config_path, model_config_path, icl_img_root
    ):
        self.icl_img_root = icl_img_root
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.icl_examples = self._init_icl_examples(icl_config_path)
        self.client = Client(model_config_path, model_name)

    @staticmethod
    def _load_image_as_tensor(path):
        # Load image from disk and convert to torch tensor in [0,1]
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)

    def _get_full_path(self, path):
        return f"{self.icl_img_root}/{path}"

    def _init_icl_examples(self, icl_config_path):
        icl_examples = []
        icl_dicts = load_yaml(icl_config_path)
        for icl_dict in icl_dicts:
            icl_dict_tensor = {}
            for key in icl_dict:
                if key != "rationale":
                    icl_dict_tensor[key] = GPTScorer._load_image_as_tensor(
                        self._get_full_path(icl_dict[key])
                    )
                else:
                    icl_dict_tensor[key] = icl_dict[key]
            icl_examples.append(ScoringExample(**icl_dict_tensor))
        return icl_examples

    @staticmethod
    def encode_image_tensor(img_tensor: torch.Tensor) -> str:
        img = img_tensor.detach().cpu().clamp(0, 1)
        # 2) convert to PIL Image
        pil_img = TF.to_pil_image(img)
        # 3) save to a bytes buffer as PNG
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        # 4) base64‐encode and return UTF‐8 string
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def wrap_text(text):
        return {"type": "text", "text": text}

    @staticmethod
    def wrap_encoded_img(img_base64):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        }

    def _build_image_content(self, example):
        """Build image content for reconstruction evaluation.

        Args:
            example: ScoringExample containing image data
        """
        base64_image_input_full = GPTScorer.encode_image_tensor(
            example.input_full_highlighted
        )
        base64_image_input_cropped = GPTScorer.encode_image_tensor(
            example.input_cropped_highlighted
        )
        base64_image_input_cropped_unmasked = GPTScorer.encode_image_tensor(
            example.input_cropped
        )

        content = [
            GPTScorer.wrap_text(
                "Below is the input image, where the input object is highlighted with a purple mask."
            ),
            GPTScorer.wrap_encoded_img(base64_image_input_full),
            GPTScorer.wrap_text(
                "Below is a close-up of the same image, centered around the input object (highlighted with a purple mask)."
            ),
            GPTScorer.wrap_encoded_img(base64_image_input_cropped),
            GPTScorer.wrap_text(
                "Below is a close-up of the same image, centered around the input object, without the highlight mask for better visibility."
            ),
            GPTScorer.wrap_encoded_img(base64_image_input_cropped_unmasked),
        ]

        rendered_input = GPTScorer.encode_image_tensor(example.rendered)
        content.extend(
            [
                GPTScorer.wrap_text(
                    "Below is the reconstruction rendered in 4 views: front, side, top, isometric."
                ),
                GPTScorer.wrap_encoded_img(rendered_input),
            ]
        )
        return content

    def query_imgs(self, test_example, prompt, build_img_content_fn):
        content = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]

        for example in self.icl_examples:
            user_contents = build_img_content_fn(example)
            content.append({"role": "user", "content": user_contents})
            content.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example.rationale}],
                }
            )

        cur_example_content = build_img_content_fn(test_example)
        content.append({"role": "user", "content": cur_example_content})

        response = self.client.query(content)
        return response

    def query_recon(self, input_dict, build_context_fn):
        out = self.query_imgs(input_dict, self.prompt, build_context_fn)
        last_line = out.strip().split("\n")[-1]
        match = re.search(r"\d+", last_line)
        value = int(match.group()) if match else None
        return value, out

    def safe_query_recon(
        self, input_dict, build_context_fn, max_retries: int = 5, base_sleep: int = 1
    ):
        """
        Query GPT-4 Vision with automatic retries.

        Raises the last exception if all retries fail.
        """
        for attempt in range(1, max_retries + 1):
            try:
                return self.query_recon(input_dict, build_context_fn)
            except Exception as e:
                if attempt == max_retries:
                    raise  # bubble up, this will be caught in compute_metrics and log out id
                sleep = base_sleep * 2 ** (attempt - 1)
                print(
                    f"[safe_query_recon] attempt {attempt}/{max_retries} failed "
                    f"({type(e).__name__}: {e}). Retrying in {sleep}s…",
                    flush=True,
                )
                time.sleep(sleep)

    def score(self, input_dict):
        input_keys = {
            "input_full_highlighted",
            "input_cropped_highlighted",
            "input_cropped",
            "rendered",
        }
        filtered_input = {k: v for k, v in input_dict.items() if k in input_keys}
        input_example = ScoringExample(**filtered_input)
        return self.safe_query_recon(input_example, self._build_image_content)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
