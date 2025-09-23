import os.path
import zipfile

import pytest
import requests
from filelock import FileLock

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
CLIP_RN50_TORCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/torch/model.pt"
CLIP_RN50_ONNX_VISUAL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/onnx/visual.onnx"
CLIP_RN50_ONNX_TEXTUAL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/onnx/textual.onnx"
PE_MODEL_URL = "https://storage.googleapis.com/roboflow-tests-assets/perception-encoder/pe-core-b16-224/model.pt"
PE_CONFIG_URL = "https://storage.googleapis.com/roboflow-tests-assets/perception-encoder/pe-core-b16-224/config.json"
FLORENCE2_BASE_FT_URL = "https://storage.googleapis.com/roboflow-tests-assets/florence2/florence-2-base-converted-for-transformers-056.zip"
FLORENCE2_LARGE_FT_URL = "https://storage.googleapis.com/roboflow-tests-assets/florence2/florence-2-large-converted-for-transformers-056.zip"
QWEN25VL_3B_FT_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/qwen/qwen25vl-3b.zip"
)
PALIGEMMA_BASE_FT_URL = "https://storage.googleapis.com/roboflow-tests-assets/paligemma/paligemma2-3b-pt-224.zip"
SMOLVLM_BASE_FT_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/smolvlm/smolvlm-256m.zip"
)
MOONDREAM2_BASE_FT_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/moondream2/moondream2-2b.zip"
)
COIN_COUNTING_RFDETR_NANO_TORCH_CS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/coin-counting-rfdetr-nano-torch-cs-stretch-640.zip"
COIN_COUNTING_RFDETR_NANO_ONNX_CS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-cs-stretch-640.zip"
COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-static-crop-letterbox-640.zip"
COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-torch-static-crop-letterbox-640.zip"

COIN_COUNTING_RFDETR_NANO_ONNX_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-center-crop-640.zip"
COIN_COUNTING_RFDETR_NANO_TORCH_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-torch-center-crop-640.zip"
COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-static-crop-center-crop-640.zip"
COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-torch-static-crop-center-crop-640.zip"
OG_RFDETR_WEIGHTS_URL = "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth"

COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-letterbox.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-letterbox-fused-nms.zip"
COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-static-bs-letterbox.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-letterbox.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-letterbox-fused-nms.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-static-crop-stretch-nms-fused.zip"
COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-static-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-static-crop-stretch-nms-fused.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-center-crop.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_NMS_FUSED_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-center-crop-fused-nms.zip"
COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-static-bs-center-crop.zip"
COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-center-crop.zip"
COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-center-crop-fused-nms.zip"

COIN_COUNTING_YOLO_NAS_ONNX_DYNAMIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-dynamic-bs-letterbox.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-letterbox.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-static-crop-letterbox.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-static-crop-center-crop.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-center-crop.zip"


@pytest.fixture(scope="module")
def original_clip_download_dir() -> str:
    clip_dir = os.path.join(MODELS_DIR, "clip_original")
    os.makedirs(clip_dir, exist_ok=True)
    return clip_dir


@pytest.fixture(scope="module")
def clip_rn50_pytorch_path() -> str:
    package_path = os.path.join(MODELS_DIR, "clip_rn50", "torch")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    _download_if_not_exists(file_path=model_path, url=CLIP_RN50_TORCH_URL)
    return package_path


@pytest.fixture(scope="module")
def clip_rn50_onnx_path() -> str:
    package_path = os.path.join(MODELS_DIR, "clip_rn50", "onnx")
    os.makedirs(package_path, exist_ok=True)
    visual_path = os.path.join(package_path, "visual.onnx")
    textual_path = os.path.join(package_path, "textual.onnx")
    _download_if_not_exists(file_path=visual_path, url=CLIP_RN50_ONNX_VISUAL)
    _download_if_not_exists(file_path=textual_path, url=CLIP_RN50_ONNX_TEXTUAL)
    return package_path


@pytest.fixture(scope="module")
def perception_encoder_path() -> str:
    package_path = os.path.join(MODELS_DIR, "perception_encoder")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    config_path = os.path.join(package_path, "config.json")
    _download_if_not_exists(file_path=model_path, url=PE_MODEL_URL)
    _download_if_not_exists(file_path=config_path, url=PE_CONFIG_URL)
    return package_path


def _download_if_not_exists(file_path: str, url: str, lock_timeout: int = 120) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock_path = f"{file_path}.lock"
    with FileLock(lock_file=lock_path, timeout=lock_timeout):
        if os.path.exists(file_path):
            return None
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


@pytest.fixture(scope="module")
def florence2_base_ft_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "florence2")
    unzipped_package_path = os.path.join(package_dir, "florence-2-base")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "base-ft.zip")
    _download_if_not_exists(file_path=zip_path, url=FLORENCE2_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=120):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def florence2_large_ft_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "florence2")
    unzipped_package_path = os.path.join(package_dir, "florence-2-base")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "large-ft.zip")
    _download_if_not_exists(file_path=zip_path, url=FLORENCE2_LARGE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=120):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def qwen25vl_3b_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "qwen25vl-3b")
    unzipped_package_path = os.path.join(package_dir, "weights")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "qwen25vl-3b.zip")
    _download_if_not_exists(file_path=zip_path, url=QWEN25VL_3B_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=120):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def paligemma_3b_224_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "paligemma2-3b-pt-224")
    unzipped_package_path = os.path.join(package_dir, "weights")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "paligemma2-3b-pt-224.zip")
    _download_if_not_exists(file_path=zip_path, url=PALIGEMMA_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=120):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def smolvlm_256m_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "smolvlm-256m")
    unzipped_package_path = os.path.join(package_dir, "weights")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "smolvlm-256m.zip")
    _download_if_not_exists(file_path=zip_path, url=SMOLVLM_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=120):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def moondream2_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "moondream2")
    unzipped_package_path = os.path.join(package_dir, "moondream2-2b")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "moondream2-2b.zip")
    _download_if_not_exists(file_path=zip_path, url=MOONDREAM2_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=120):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


def download_model_package(
    model_package_zip_url: str,
    package_name: str,
) -> str:
    package_dir = os.path.join(MODELS_DIR, package_name)
    unzipped_package_path = os.path.join(package_dir, "unpacked")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "package.zip")
    _download_if_not_exists(file_path=zip_path, url=model_package_zip_url)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=120):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzipped_package_path)
    return unzipped_package_path


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_cs_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_CS_STRETCH_URL,
        package_name="coin-counting-rfdetr-nano-torch-cs-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_cs_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_CS_STRETCH_URL,
        package_name="coin-counting-rfdetr-nano-onnx-cs-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_LETTERBOX_URL,
        package_name="coin-counting-rfdetr-nano-onnx-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_LETTERBOX_URL,
        package_name="coin-counting-rfdetr-nano-torch-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-onnx-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-torch-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-onnx-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-torch-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def og_rfdetr_base_weights() -> str:
    package_path = os.path.join(MODELS_DIR, "og-rfdetr-base")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    _download_if_not_exists(file_path=model_path, url=OG_RFDETR_WEIGHTS_URL)
    return model_path


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_letterbox_fused_nms_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_FUSED_NMS_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-letterbox-fused-nms",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolov8n-onnx-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_FUSED_NMS_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-fused-nms-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_static_crop_stretch_nms_fused_package() -> (
    str
):
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-static-crop-stretch-nms-fused",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_static_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolov8n-onnx-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package() -> (
    str
):
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-static-crop-stretch-fused-nms",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_center_crop_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_center_crop_fused_nms_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_NMS_FUSED_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-center-crop-fused-nms",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_static_bs_center_crop_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov8n-onnx-static-bs-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_toch_script_static_bs_center_crop_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_FUSED_NMS_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-center-fused-nms-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_dynamic_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_DYNAMIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolo-nas-dynamic-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolo-nas-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_LETTERBOX_URL,
        package_name="coin-counting-yolo-nas-static-bs-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolo-nas-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL,
        package_name="coin-counting-yolo-nas-static-bs-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolo-nas-static-bs-center-crop",
    )
