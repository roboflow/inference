"""Reference `cosmos_anomalygen_runtime.py` for the Cosmos AnomalyGen package.

This is the self-contained runtime module that ships INSIDE the
`cosmos-anomalygen` model package (injected by pull_anomalygen_weights.py
--runtime-module). `CosmosAnomalyGen.from_pretrained` imports the
`CosmosAnomalyGenRuntime` class from it via `import_class_from_file`.

Unlike the world tower (which rides diffusers), AnomalyGen has no released
pipeline: this runtime drives NVIDIA's GA `paidf-anomalygen` stack directly
(`cosmos_predict2` + `imaginaire`), so it must run inside an environment where
that repo is importable - in practice a container built on the published GA
image (`paidf-anomalygen:ga`). The heavy imports are lazy so the module itself
imports anywhere.

Package layout expected at `load(package_dir)` (all paths package-relative):

    cosmos_anomalygen_runtime.py       this file
    inference_config.json              {"experiment": ..., "guidance": ..., "anomaly_types": [...]}
    class_names.txt                    one trained "<category>+<class>" anomaly type per line
    ag_config.yaml                     frozen training config (the run dir's copy)
    checkpoints/model/iter_XXXXXXXXX.pt        trained adapter + anomaly embeddings (~14 MB)
    checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt          frozen DiT
    checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/tokenizer/tokenizer.pth   frozen VAE
    checkpoints/google-t5/t5-large/...          frozen text encoder
    checkpoints/NVDINOV2/nv_dinov2_classification_model.ckpt           frozen mask encoder
    checkpoints/facebook/dinov2-large/...       correspondence backbone (prefetched at init)

The trained artifact and the frozen base towers share one `checkpoints/` tree
on purpose: the GA loader reads `<ckpt_dir>/checkpoints/model/iter_*.pt` for
the adapter while the base-weight paths are overridden here to absolute
package paths, so the package needs no particular working directory.

Contract expected by CosmosAnomalyGen (images RGB numpy arrays, masks uint8
0/255 at image resolution):
- load(package_dir, device) -> runtime
- generate(image, mask, anomaly_type, guidance, num_steps, seed, num_images,
  crop_and_paste, crop_ratio, poisson_blend) -> [RGB arrays]

Generation reuses the GA SDG entry path 1:1 (temp one-line JSONL ->
`AnomalyInpaintDataset` -> `AnomalyInpaintCondition` -> `inpaint_image`), so
preprocessing, RePaint-style latent replacement, crop/paste and PSNR reporting
stay byte-compatible with `scripts.anomaly_gen.synthetic_dataset_generation`.
"""

import glob
import importlib
import json
import os
import re
import tempfile
from typing import List, Optional

import numpy as np
import torch

DEFAULT_EXPERIMENT = "predict2_anomaly_gen_ddp_2b"
_DIT_PATH = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
_VAE_PATH = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/tokenizer/tokenizer.pth"
_T5_PATH = "checkpoints/google-t5/t5-large"
_NVDINOV2_PATH = "checkpoints/NVDINOV2/nv_dinov2_classification_model.ckpt"
_CORRESPONDENCE_PATH = "checkpoints/facebook/dinov2-large"


class CosmosAnomalyGenRuntime:

    @classmethod
    def load(cls, package_dir: str, device: torch.device) -> "CosmosAnomalyGenRuntime":
        try:
            import yaml
            from cosmos_predict2.inference.anomaly_gen.initialize import (
                initialize_anomaly_diffusion_model,
            )
            from imaginaire.utils.config_helper import get_config_module, override
            from scripts.anomaly_gen.ag_train import set_nested_attributes
        except ImportError as exc:
            raise RuntimeError(
                "Cosmos AnomalyGen requires NVIDIA's paidf-anomalygen stack "
                "(cosmos_predict2 + imaginaire) on the python path. Run inside "
                "a container built on the paidf-anomalygen:ga base image."
            ) from exc

        package_dir = os.path.abspath(package_dir)
        inference_config = _read_optional_json(
            os.path.join(package_dir, "inference_config.json")
        )
        anomaly_types = _read_anomaly_types(package_dir, inference_config)
        step = _find_checkpoint_step(package_dir)

        config_module = get_config_module("cosmos_predict2/configs/base/ag_config.py")
        config = importlib.import_module(config_module).make_config()
        experiment = inference_config.get("experiment", DEFAULT_EXPERIMENT)
        config = override(config, ["--", f"experiment={experiment}"])

        ag_config_path = os.path.join(package_dir, "ag_config.yaml")
        with open(ag_config_path) as fp:
            ag_config = yaml.safe_load(fp)
        set_nested_attributes(config, ag_config)

        # The GA configs address every frozen tower relative to the repo
        # checkout; repoint them into the package so no working directory or
        # repo-local `checkpoints/` tree is assumed.
        set_nested_attributes(
            config,
            {
                "model": {
                    "config": {
                        # The GA model __init__ prefetches the correspondence
                        # backbone unconditionally (early-stop metric), so it
                        # must resolve inside the package too.
                        "correspondence_backbone": os.path.join(
                            package_dir, _CORRESPONDENCE_PATH
                        ),
                        "model_manager_config": {
                            "dit_path": os.path.join(package_dir, _DIT_PATH),
                        },
                        "pipe_config": {
                            "tokenizer": {
                                "vae_pth": os.path.join(package_dir, _VAE_PATH),
                            },
                            "guardrail_config": {
                                "checkpoint_dir": os.path.join(
                                    package_dir, "checkpoints"
                                ),
                            },
                        },
                        "ag_config": {
                            "t5_model_name": os.path.join(package_dir, _T5_PATH),
                            "mask_encoder": {
                                "encoder_config": {
                                    "init_cfg": {
                                        "checkpoint": os.path.join(
                                            package_dir, _NVDINOV2_PATH
                                        ),
                                    }
                                }
                            },
                        },
                    }
                }
            },
        )
        if getattr(config.model.config, "fsdp_shard_size", 0) != 0:
            config.model.config.fsdp_shard_size = 0

        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        model = initialize_anomaly_diffusion_model(config, package_dir, step)
        return cls(model=model, device=device, anomaly_types=anomaly_types)

    def __init__(self, model, device: torch.device, anomaly_types: List[str]):
        self._model = model
        self._device = device
        self._anomaly_types = anomaly_types

    @property
    def anomaly_types(self) -> List[str]:
        return list(self._anomaly_types)

    def generate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        anomaly_type: str,
        guidance: float,
        num_steps: int,
        seed: int,
        num_images: int,
        crop_and_paste: bool,
        crop_ratio: Optional[float],
        poisson_blend: bool,
    ) -> List[np.ndarray]:
        from cosmos_predict2.data.anomaly_gen.anomaly_dataset import (
            AnomalyInpaintDataset,
        )
        from cosmos_predict2.inference.anomaly_gen.inference_anomaly_diffusion_utils import (
            inpaint_image,
        )
        from cosmos_predict2.inference.anomaly_gen.inpaint_condition import (
            AnomalyInpaintCondition,
        )
        from imaginaire.utils import misc
        from PIL import Image

        if self._anomaly_types and anomaly_type not in self._anomaly_types:
            raise ValueError(
                f"Unknown anomaly type {anomaly_type!r}; this package was "
                f"trained on: {', '.join(self._anomaly_types)}"
            )

        with tempfile.TemporaryDirectory(prefix="anomalygen-") as workdir:
            image_path = os.path.join(workdir, "image.png")
            mask_path = os.path.join(workdir, "mask.png")
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            entry = {
                "image_filename": image_path,
                "mask_filename": mask_path,
                "anomaly_type": anomaly_type,
                "guidance": guidance,
                "num_steps": num_steps,
                "crop_and_paste": crop_and_paste,
                "num_generated_images": num_images,
                "poisson_blend": poisson_blend,
                "iteration_generation_max_instance": 1,
                "index": 0,
            }
            if crop_and_paste and crop_ratio is not None:
                entry["crop_ratio"] = crop_ratio
            jsonl_path = os.path.join(workdir, "testcase.jsonl")
            with open(jsonl_path, "w") as fp:
                fp.write(json.dumps(entry) + "\n")

            # The dataset applies the same defaulting / mask handling as the
            # SDG script, so one JSONL entry here behaves exactly like one
            # line fed to `synthetic_dataset_generation`.
            dataset = AnomalyInpaintDataset(jsonl_path)
            batch = dataset._collate_fn([dataset[0]])
            batch["seed"] = seed
            condition = AnomalyInpaintCondition(**batch)

            misc.set_random_seed(seed, by_rank=False)
            with torch.no_grad():
                inpainting_result, _, _ = inpaint_image(condition, self._model)

        reconstructed = inpainting_result["reconstructed_image"]
        return [np.asarray(item.convert("RGB")) for item in reconstructed]


def _read_optional_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as fp:
        return json.load(fp)


def _read_anomaly_types(package_dir: str, inference_config: dict) -> List[str]:
    class_names_path = os.path.join(package_dir, "class_names.txt")
    if os.path.exists(class_names_path):
        with open(class_names_path) as fp:
            return [line.strip() for line in fp if line.strip()]
    return list(inference_config.get("anomaly_types", []))


def _find_checkpoint_step(package_dir: str) -> int:
    pattern = os.path.join(package_dir, "checkpoints", "model", "iter_*.pt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No trained checkpoint found under {pattern}; the package must "
            "contain the fine-tuned adapter as checkpoints/model/iter_<step>.pt"
        )
    match = re.search(r"iter_(\d+)\.pt$", candidates[-1])
    return int(match.group(1))
