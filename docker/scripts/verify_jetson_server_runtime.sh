#!/bin/sh
set -eu

/usr/local/cuda/bin/ptxas --version >/dev/null
test -f /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
test -f /usr/local/share/licenses/cuda-ptxas
test "${TRITON_PTXAS_PATH}" = /usr/local/cuda/bin/ptxas
test "${TRITON_PTXAS_BLACKWELL_PATH}" = /usr/local/cuda/bin/ptxas
test ! -e /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas
test ! -e /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas-blackwell
test -x /usr/local/bin/inference
test -x /usr/local/bin/run_uvicorn.sh
test -x /usr/local/bin/verify_tensor_tracking_runtime
test -s /opt/roboflow/lib/libroboflow_jetson_tensor.so.1
test -f /usr/include/python3.12/Python.h
command -v gcc >/dev/null
command -v python3-config >/dev/null
DISABLE_VERSION_CHECK=true inference --help >/dev/null
torchvision_image="$(
    python3 -c 'from torchvision._internally_replaced_utils import _get_extension_path; print(_get_extension_path("image"))'
)"
ldd "${torchvision_image}" | grep -q 'libnvjpeg.so.13'

for command_name in \
    c++ \
    cmake \
    curl \
    g++ \
    git \
    make \
    meson \
    ninja \
    nvcc \
    pip \
    pip3; do
    if command -v "${command_name}" >/dev/null 2>&1; then
        echo "Build command present in runtime: ${command_name}" >&2
        exit 1
    fi
done

for package in \
    build-essential \
    cmake \
    cuda-nvcc-13-2 \
    cuda-toolkit-13-2 \
    g++ \
    make \
    ninja-build \
    pkg-config; do
    if dpkg-query -W -f='${db:Status-Status}' "${package}" 2>/dev/null |
        grep -qx installed; then
        echo "Build dependency present in runtime: ${package}" >&2
        exit 1
    fi
done

development_packages="$(
    dpkg-query -W -f='${binary:Package}\n' |
        grep -E -- '-dev(:[^[:space:]]+)?$' || true
)"
for package in ${development_packages}; do
    package="${package%%:*}"
    case "${package}" in
        libc6-dev | libcrypt-dev | libexpat1-dev | libgcc-13-dev | \
            libpython3.12-dev | linux-libc-dev | python3.12-dev | zlib1g-dev)
            ;;
        *)
            echo "Development package present in runtime: ${package}" >&2
            exit 1
            ;;
    esac
done

python3 - <<'PY'
import cv2
import flash_attn
import importlib.metadata
import importlib.util
import onnxruntime
import supervision
import tensorrt
import torch
import torchvision
import triton
import trackers
from torchvision import io as torchvision_io

arch_flags = set(torch._C._cuda_getArchFlags().split())
assert {"sm_87", "sm_110"}.issubset(arch_flags), arch_flags

providers = set(onnxruntime.get_available_providers())
assert {"CUDAExecutionProvider", "TensorrtExecutionProvider"}.issubset(providers)

assert cv2.cuda.getCudaEnabledDeviceCount() >= 0
assert flash_attn.__version__
assert tensorrt.__version__
assert torch.__version__
assert torchvision.__version__
assert triton.__version__
assert torchvision_io
assert hasattr(torch.ops.image, "decode_jpegs_cuda")
assert importlib.metadata.version("superiorvision") == "0.30.0.dev2"
assert importlib.metadata.version("tracktors") == "2.6.0.dev1"
assert supervision.__file__
assert trackers.__file__
assert cv2.__file__.startswith("/opt/opencv/python/"), cv2.__file__

for replaced_distribution in ("supervision", "trackers"):
    try:
        installed_version = importlib.metadata.version(replaced_distribution)
    except importlib.metadata.PackageNotFoundError:
        continue
    raise AssertionError((replaced_distribution, installed_version))

torch_requirements = importlib.metadata.requires("torch") or []
assert not any(
    requirement.partition(";")[0].strip().lower().startswith("nvidia-")
    for requirement in torch_requirements
), torch_requirements

pip_cuda_distributions = sorted(
    name
    for distribution in importlib.metadata.distributions()
    if (name := distribution.metadata.get("Name"))
    and (
        name.lower().startswith("nvidia-")
        or name.lower() in {"cuda-bindings", "cuda-toolkit"}
    )
)
assert not pip_cuda_distributions, pip_cuda_distributions

for module in (
    "build",
    "cmake",
    "expecttest",
    "hypothesis",
    "lintrunner",
    "lit",
    "ninja",
    "parameterized",
    "pydocstyle",
    "twine",
):
    assert importlib.util.find_spec(module) is None, module
PY

extension_directory="$(mktemp -d)"
trap 'rm -rf "${extension_directory}"' EXIT
cat > "${extension_directory}/runtime_probe.c" <<'C'
#include <Python.h>

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "runtime_probe",
    NULL,
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit_runtime_probe(void) {
    return PyModule_Create(&module);
}
C
python_include="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
gcc -O2 -shared -fPIC \
    -I"${python_include}" \
    "${extension_directory}/runtime_probe.c" \
    -o "${extension_directory}/runtime_probe$(python3-config --extension-suffix)"
PYTHONPATH="${extension_directory}" python3 -c "import runtime_probe"
