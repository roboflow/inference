import os
import shutil

import numpy as np
import pytest
from PIL import Image

from inference.core.env import MODEL_CACHE_DIR


api_key = os.environ.get("API_KEY")


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.fixture(scope="function")
def clean_smolvlm_cache():
    cache_dir = os.path.join(MODEL_CACHE_DIR, "smolvlm2")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    yield


@pytest.fixture(scope="function")
def example_pil_image() -> Image.Image:
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SMOLVLM_TEST", True))
    or bool_env(os.getenv("SKIP_LMM_TEST", True))
    or not api_key,
    reason="Skipping SmolVLM download test",
)
def test_smolvlm_model_download_and_inference(
    clean_smolvlm_cache, example_pil_image
) -> None:
    from inference import get_model

    model = get_model("smolvlm2", api_key=api_key)
    result = model.predict(example_pil_image, prompt="What is in this image?")

    assert result is not None
    assert len(result) > 0
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
