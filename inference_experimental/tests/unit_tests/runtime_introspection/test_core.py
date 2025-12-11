from subprocess import CompletedProcess
from unittest import mock
from unittest.mock import MagicMock, mock_open

import pytest
from inference_exp.errors import JetsonTypeResolutionError, RuntimeIntrospectionError
from inference_exp.runtime_introspection import core
from inference_exp.runtime_introspection.core import (
    ensure_jetson_l4t_declared_for_jetson_hardware,
    get_available_gpu_devices,
    get_available_gpu_devices_cc,
    get_cuda_version,
    get_driver_version,
    get_jetson_type,
    get_jetson_type_from_device_tree,
    get_jetson_type_from_hardware_inspection,
    get_l4t_version,
    get_l4t_version_from_tegra_release,
    get_onnxruntime_info,
    get_os_version,
    get_torch_version,
    get_torchvision_version,
    get_trt_version,
    get_trt_version_from_libnvinfer,
    is_hf_transformers_available,
    is_running_on_jetson,
    is_torch_available,
    is_trt_python_package_available,
    is_ultralytics_available,
    resolve_jetson_type,
)
from packaging.version import Version

"""I mock your mocks :)"""


@mock.patch.object(core, "RUNNING_ON_JETSON", "true")
def test_is_running_on_jetson_when_env_variable_is_set() -> None:
    # when
    is_running_on_jetson.cache_clear()
    get_jetson_type.cache_clear()
    try:
        result = is_running_on_jetson()
    finally:
        is_running_on_jetson.cache_clear()
        get_jetson_type.cache_clear()

    # then
    assert result is True


@mock.patch.object(core, "get_jetson_type")
@mock.patch.object(core, "RUNNING_ON_JETSON", None)
def test_is_running_on_jetson_when_get_jetson_type_fallback_is_needed(
    get_jetson_type_mock: MagicMock,
) -> None:
    # given
    get_jetson_type_mock.return_value = "jetson-orin-nx"

    # when
    is_running_on_jetson.cache_clear()
    get_jetson_type.cache_clear()
    try:
        result = is_running_on_jetson()
    finally:
        is_running_on_jetson.cache_clear()
        get_jetson_type.cache_clear()

    # then
    assert result is True


@mock.patch.object(core, "get_jetson_type")
@mock.patch.object(core, "RUNNING_ON_JETSON", None)
def test_is_running_on_jetson_when_cannot_determine_jetson_characteristics(
    get_jetson_type_mock: MagicMock,
) -> None:
    # given
    get_jetson_type_mock.return_value = None

    # when
    is_running_on_jetson.cache_clear()
    get_jetson_type.cache_clear()
    try:
        result = is_running_on_jetson()
    finally:
        is_running_on_jetson.cache_clear()
        get_jetson_type.cache_clear()

    # then
    assert result is False


@mock.patch.object(core, "torch")
def test_get_available_gpu_devices(torch_mock: MagicMock) -> None:
    # given
    torch_mock.cuda.device_count.return_value = 2
    torch_mock.cuda.get_device_name.side_effect = ["Tesla T4", "Nvidia L4"]
    get_available_gpu_devices.cache_clear()

    # when
    try:
        result = get_available_gpu_devices()
    finally:
        get_available_gpu_devices.cache_clear()

    # then
    assert result == ["tesla-t4", "nvidia-l4"]


@mock.patch.object(core, "torch")
def test_get_available_gpu_devices_cc(torch_mock: MagicMock) -> None:
    # given
    torch_mock.cuda.device_count.return_value = 2
    torch_mock.cuda.get_device_capability.side_effect = [(7, 5), (8, 9)]
    get_available_gpu_devices_cc.cache_clear()

    # when
    try:
        result = get_available_gpu_devices_cc()
    finally:
        get_available_gpu_devices_cc.cache_clear()

    # then
    assert result == [Version("7.5"), Version("8.9")]


@mock.patch.object(core, "subprocess")
def test_get_cuda_version_when_sub_process_raise_error(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.side_effect = Exception()
    get_cuda_version.cache_clear()

    # when
    try:
        result = get_cuda_version()
    finally:
        get_cuda_version.cache_clear()

    # then
    assert result is None


@mock.patch.object(core, "subprocess")
def test_get_cuda_version_when_sub_process_return_error_code(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=1,
    )
    get_cuda_version.cache_clear()

    # when
    try:
        result = get_cuda_version()
    finally:
        get_cuda_version.cache_clear()

    # then
    assert result is None


@mock.patch.object(core, "subprocess")
def test_get_cuda_version_when_sub_process_return_not_parsable_output(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(), returncode=0, stdout="This is some\nInvalid output"
    )
    get_cuda_version.cache_clear()

    # when
    try:
        result = get_cuda_version()
    finally:
        get_cuda_version.cache_clear()

    # then
    assert result is None


@mock.patch.object(core, "subprocess")
def test_get_cuda_version_when_sub_process_return_parsable_output(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=0,
        stdout="""
ii  cuda-cudart-12-6                           12.6.68-1                                   arm64        CUDA Runtime native Libraries
ii  cuda-cudart-dev-12-6                       12.6.68-1                                   arm64        CUDA Runtime native dev links, headers
""",
    )
    get_cuda_version.cache_clear()

    # when
    try:
        result = get_cuda_version()
    finally:
        get_cuda_version.cache_clear()

    # then
    assert result == Version("12.6.68.post1")


@mock.patch.object(core, "subprocess")
def test_get_trt_version_when_libnvifer_found(subprocess_mock: MagicMock) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=0,
        stdout="""
ii  libnvinfer-bin                             10.3.0.30-1+cuda12.5                        arm64        TensorRT binaries
""",
    )
    get_trt_version.cache_clear()
    get_trt_version_from_libnvinfer.cache_clear()

    # when
    try:
        result = get_trt_version()
    finally:
        get_trt_version.cache_clear()
        get_trt_version_from_libnvinfer.cache_clear()

    # then
    assert result == Version("10.3.0.30-1")


@mock.patch.object(core, "subprocess")
def test_get_trt_version_when_libnvifer_not_found_and_trt_cannot_be_imported(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=1,
    )
    get_trt_version.cache_clear()
    get_trt_version_from_libnvinfer.cache_clear()

    # when
    try:
        result = get_trt_version()
    finally:
        get_trt_version.cache_clear()
        get_trt_version_from_libnvinfer.cache_clear()

    # then
    assert result is None


@mock.patch.object(core, "subprocess")
def test_get_trt_version_when_libnvifer_not_found_and_trt_can_be_imported(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=1,
    )
    get_trt_version.cache_clear()
    get_trt_version_from_libnvinfer.cache_clear()

    try:
        with mock.patch("builtins.__import__") as mocked_import:
            trt_module_mock = MagicMock()
            trt_module_mock.__version__ = "10.2.0.1"
            mocked_import.return_value = trt_module_mock

            # when
            result = get_trt_version()
    finally:
        get_trt_version.cache_clear()
        get_trt_version_from_libnvinfer.cache_clear()

    # then
    assert result == Version("10.2.0.1")


@mock.patch.object(core, "subprocess")
def test_get_trt_version_from_libnvinfer_when_output_cannot_be_parsed(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=0,
        stdout="Invalid output",
    )

    get_trt_version_from_libnvinfer.cache_clear()

    # when
    try:
        result = get_trt_version_from_libnvinfer()
    finally:
        get_trt_version_from_libnvinfer.cache_clear()

    # then
    assert result is None


@mock.patch.object(core, "os")
def test_get_jetson_type_when_jetson_module_delivered_in_env(
    os_mock: MagicMock,
) -> None:
    # given
    os_mock.getenv.return_value = "NVIDIA Jetson Orin Nano (Developer kit)"

    # when
    get_jetson_type.cache_clear()
    try:
        result = get_jetson_type()
    finally:
        get_jetson_type.cache_clear()

    # then
    assert result == "nvidia-jetson-orin-nano"


@mock.patch.object(core, "subprocess")
@mock.patch.object(core, "os")
def test_get_jetson_type_when_jetson_module_from_hardware_introspection(
    os_mock: MagicMock,
    subprocess_mock: MagicMock,
) -> None:
    # given
    os_mock.getenv.return_value = None
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=0,
        stdout="""
    product: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
       product: NVIDIA Jetson Orin Nano HDA HDMI/DP,pcm=3
       product: NVIDIA Jetson Orin Nano HDA HDMI/DP,pcm=7
       product: NVIDIA Jetson Orin Nano HDA HDMI/DP,pcm=8
       product: NVIDIA Jetson Orin Nano HDA HDMI/DP,pcm=9
""",
    )

    # when
    get_jetson_type.cache_clear()
    try:
        result = get_jetson_type()
    finally:
        get_jetson_type.cache_clear()

    # then
    assert result == "nvidia-jetson-orin-nano"


@mock.patch.object(core, "subprocess")
@mock.patch.object(core, "os")
def test_get_jetson_type_when_jetson_module_from_device_tree(
    os_mock: MagicMock,
    subprocess_mock: MagicMock,
) -> None:
    # given
    os_mock.getenv.return_value = None
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=1,
    )
    mocked_open_results = mock_open(
        read_data="NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"
    )

    # when
    get_jetson_type.cache_clear()
    try:
        with mock.patch("builtins.open", mocked_open_results):
            result = get_jetson_type()
    finally:
        get_jetson_type.cache_clear()

    # then
    assert result == "nvidia-jetson-orin-nano"


@mock.patch.object(core, "subprocess")
def test_get_jetson_type_from_hardware_inspection_when_content_is_not_parsable(
    subprocess_mock: MagicMock,
) -> None:
    # given
    subprocess_mock.run.return_value = CompletedProcess(
        args=(),
        returncode=0,
        stdout="Invalid content",
    )

    # when
    result = get_jetson_type_from_hardware_inspection()

    # then
    assert result is None


def test_get_jetson_type_from_device_tree_when_the_content_is_not_parsable() -> None:
    # given
    mocked_open_results = mock_open(read_data="Dummy")

    # when
    with mock.patch("builtins.open", mocked_open_results):
        result = get_jetson_type_from_device_tree()

    # then
    assert result is None


@pytest.mark.parametrize(
    "jetson_module_name, expected_result",
    [
        (
            "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
            "nvidia-jetson-orin-nano",
        ),
        ("NVIDIA Jetson Orin Nano", "nvidia-jetson-orin-nano"),
        (
            "NVIDIA Jetson Orin NX Engineering Reference Developer Kit",
            "nvidia-jetson-orin-nx",
        ),
        (
            "NVIDIA Jetson Orin NX Engineering",
            "nvidia-jetson-orin-nx",
        ),
        (
            "NVIDIA Jetson AGX Orin Developer Kit",
            "nvidia-jetson-agx-orin",
        ),
        (
            "NVIDIA Jetson AGX Orin",
            "nvidia-jetson-agx-orin",
        ),
        (
            "NVIDIA Jetson IGX Orin",
            "nvidia-jetson-igx-orin",
        ),
    ],
)
def test_resolve_jetson_type_when_resolution_should_be_successful(
    jetson_module_name: str, expected_result: str
) -> None:
    # when
    result = resolve_jetson_type(jetson_module_name=jetson_module_name)

    # then
    assert result == expected_result


def test_resolve_jetson_type_when_invalid_type_provided() -> None:
    # when
    with pytest.raises(JetsonTypeResolutionError):
        _ = resolve_jetson_type(jetson_module_name="invalid")


@mock.patch.object(core, "L4T_VERSION", "36.4.0")
def test_get_l4t_version_when_env_variable_set() -> None:
    # when
    get_l4t_version.cache_clear()
    try:
        result = get_l4t_version()
    finally:
        get_l4t_version.cache_clear()

    # then
    assert result == Version("36.4.0")


@mock.patch.object(core, "L4T_VERSION", None)
def test_get_l4t_version_when_env_variable_not_set() -> None:
    # given
    mocked_open_results = mock_open(
        read_data="""# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
"""
    )
    get_l4t_version.cache_clear()
    try:
        with mock.patch("builtins.open", mocked_open_results):
            # when
            result = get_l4t_version()
    finally:
        get_l4t_version.cache_clear()

    # then
    assert result == Version("36.4.3")


def test_get_l4t_version_from_tegra_release_when_the_output_is_not_parsable() -> None:
    # given
    mocked_open_results = mock_open(read_data="invalid")

    with mock.patch("builtins.open", mocked_open_results):
        # when
        result = get_l4t_version_from_tegra_release()

    # then
    assert result is None


def test_get_l4t_version_from_tegra_release_when_file_not_found() -> None:
    # given
    with mock.patch("builtins.open") as open_mock:
        open_mock.side_effect = OSError()

        # when
        result = get_l4t_version_from_tegra_release()

    # then
    assert result is None


@mock.patch.object(core, "platform")
def test_get_os_version_for_linux_when_etc_release_available(
    platform_mock: MagicMock,
) -> None:
    # given
    platform_mock.system.return_value = "Linux"
    get_os_version.cache_clear()
    mocked_open_results = mock_open(
        read_data="""
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy"""
    )
    try:
        with mock.patch("builtins.open", mocked_open_results):
            # when
            result = get_os_version()
    finally:
        get_os_version.cache_clear()

    # then
    assert result == "ubuntu-22.04"


@mock.patch.object(core, "platform")
def test_get_os_version_for_linux_when_etc_release_not_parsable(
    platform_mock: MagicMock,
) -> None:
    # given
    platform_mock.system.return_value = "Linux"
    platform_mock.platform.return_value = "linux-5.15.148-tegra-aarch64-with-glibc2.35"
    get_os_version.cache_clear()
    mocked_open_results = mock_open(read_data="invalid")
    try:
        with mock.patch("builtins.open", mocked_open_results):
            # when
            result = get_os_version()
    finally:
        get_os_version.cache_clear()

    # then
    assert result == "linux-5.15.148-tegra-aarch64-with-glibc2.35"


@mock.patch.object(core, "platform")
def test_get_os_version_for_darwin(platform_mock: MagicMock) -> None:
    # given
    platform_mock.system.return_value = "Darwin"
    platform_mock.platform.return_value = "macos-15.5-arm64-arm-64bit"
    get_os_version.cache_clear()
    try:
        # when
        result = get_os_version()
    finally:
        get_os_version.cache_clear()

    # then
    assert result == "macos-15.5-arm64-arm-64bit"


@mock.patch.object(core, "platform")
def test_get_os_version_for_windows(platform_mock: MagicMock) -> None:
    # given
    platform_mock.system.return_value = "Windows"
    platform_mock.platform.return_value = "Windows-10-10.0.19045-SP0"
    get_os_version.cache_clear()
    try:
        # when
        result = get_os_version()
    finally:
        get_os_version.cache_clear()

    # then
    assert result == "windows-10-10.0.19045-sp0"


def test_get_driver_version_when_file_not_found() -> None:
    # given
    get_driver_version.cache_clear()
    try:
        with mock.patch("builtins.open") as open_mock:
            open_mock.side_effect = FileNotFoundError()
            # when
            result = get_driver_version()
    finally:
        get_driver_version.cache_clear()

    # then
    assert result is None


def test_get_driver_version_when_file_not_parsable() -> None:
    # given
    get_driver_version.cache_clear()
    mocked_open_results = mock_open(read_data="invalid")
    try:
        with mock.patch("builtins.open", mocked_open_results):
            # when
            result = get_driver_version()
    finally:
        get_driver_version.cache_clear()

    # then
    assert result is None


def test_get_driver_version_when_file_parsable() -> None:
    # given
    get_driver_version.cache_clear()
    mocked_open_results = mock_open(
        read_data="""NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  540.4.0  Release Build  (buildbrain@mobile-u64-6336-d8000)  Tue Jan  7 17:35:12 PST 2025
GCC version:  collect2: error: ld returned 1 exit status    
"""
    )
    try:
        with mock.patch("builtins.open", mocked_open_results):
            # when
            result = get_driver_version()
    finally:
        get_driver_version.cache_clear()

    # then
    assert result == Version("540.4.0")


def test_is_trt_python_package_available_when_trt_cannot_be_imported() -> None:
    # given
    is_trt_python_package_available.cache_clear()
    try:
        with mock.patch("builtins.__import__") as mocked_import:
            mocked_import.side_effect = ImportError()
            # when
            result = is_trt_python_package_available()
    finally:
        is_trt_python_package_available.cache_clear()

    # then
    assert result is False


def test_is_trt_python_package_available_when_trt_can_be_imported() -> None:
    # given
    is_trt_python_package_available.cache_clear()
    try:
        with mock.patch("builtins.__import__"):
            # when
            result = is_trt_python_package_available()
    finally:
        is_trt_python_package_available.cache_clear()

    # then
    assert result is True


def test_is_torch_available_available_when_trt_cannot_be_imported() -> None:
    # given
    is_torch_available.cache_clear()
    try:
        with mock.patch("builtins.__import__") as mocked_import:
            mocked_import.side_effect = ImportError()
            # when
            result = is_torch_available()
    finally:
        is_torch_available.cache_clear()

    # then
    assert result is False


def test_is_torch_available_available_when_trt_can_be_imported() -> None:
    # given
    is_torch_available.cache_clear()
    try:
        with mock.patch("builtins.__import__"):
            # when
            result = is_torch_available()
    finally:
        is_torch_available.cache_clear()

    # then
    assert result is True


def test_is_hf_transformers_available_when_trt_cannot_be_imported() -> None:
    # given
    is_hf_transformers_available.cache_clear()
    try:
        with mock.patch("builtins.__import__") as mocked_import:
            mocked_import.side_effect = ImportError()
            # when
            result = is_hf_transformers_available()
    finally:
        is_hf_transformers_available.cache_clear()

    # then
    assert result is False


def test_is_hf_transformers_available_when_trt_can_be_imported() -> None:
    # given
    is_hf_transformers_available.cache_clear()
    try:
        with mock.patch("builtins.__import__"):
            # when
            result = is_hf_transformers_available()
    finally:
        is_hf_transformers_available.cache_clear()

    # then
    assert result is True


def test_is_ultralytics_available_when_trt_cannot_be_imported() -> None:
    # given
    is_ultralytics_available.cache_clear()
    try:
        with mock.patch("builtins.__import__") as mocked_import:
            mocked_import.side_effect = ImportError()
            # when
            result = is_ultralytics_available()
    finally:
        is_ultralytics_available.cache_clear()

    # then
    assert result is False


def test_is_ultralytics_available_when_trt_can_be_imported() -> None:
    # given
    is_ultralytics_available.cache_clear()
    try:
        with mock.patch("builtins.__import__"):
            # when
            result = is_ultralytics_available()
    finally:
        is_ultralytics_available.cache_clear()

    # then
    assert result is True


def test_get_onnxruntime_info_when_onnxruntime_not_available() -> None:
    # given
    get_onnxruntime_info.cache_clear()
    try:
        with mock.patch("builtins.__import__") as mocked_import:
            mocked_import.side_effect = ImportError()
            # when
            result = get_onnxruntime_info()
    finally:
        get_onnxruntime_info.cache_clear()

    # then
    assert result is None


def test_get_onnxruntime_info_when_onnxruntime_available() -> None:
    # given
    get_onnxruntime_info.cache_clear()
    try:
        with mock.patch("builtins.__import__") as import_mock:
            onnxruntime_mock = MagicMock()
            onnxruntime_mock.__version__ = "1.21.0"
            onnxruntime_mock.get_available_providers.return_value = [
                "CoreMLExecutionProvider",
                "AzureExecutionProvider",
                "CPUExecutionProvider",
            ]
            import_mock.return_value = onnxruntime_mock

            # when
            result = get_onnxruntime_info()
    finally:
        get_onnxruntime_info.cache_clear()

    # then
    assert result == (
        Version("1.21.0"),
        ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"],
    )


def test_get_torch_version_when_torch_not_available() -> None:
    # given
    get_torch_version.cache_clear()
    try:
        with mock.patch("builtins.__import__") as import_mock:
            import_mock.side_effect = ImportError()

            # when
            result = get_torch_version()
    finally:
        get_torch_version.cache_clear()

    # then
    assert result is None


def test_get_torch_version_when_torch_available_but_version_not_parsable() -> None:
    # given
    get_torch_version.cache_clear()
    try:
        with mock.patch("builtins.__import__") as import_mock:
            torch_mock = MagicMock()
            torch_mock.__version__ = "invalid"
            import_mock.return_value = torch_mock

            # when
            result = get_torch_version()
    finally:
        get_torch_version.cache_clear()

    # then
    assert result is None


@pytest.mark.parametrize(
    "version, expected_result",
    [
        ("2.6.0", Version("2.6.0")),
        ("2.6.0+cu118", Version("2.6.0")),
        ("2.6.0+cu126", Version("2.6.0")),
        ("2.6.0++gitabcdef1", Version("2.6.0")),
    ],
)
def test_get_torch_version_when_torch_available_but_version_parsable(
    version: str, expected_result: Version
) -> None:
    # given
    get_torch_version.cache_clear()
    try:
        with mock.patch("builtins.__import__") as import_mock:
            torch_mock = MagicMock()
            torch_mock.__version__ = version
            import_mock.return_value = torch_mock

            # when
            result = get_torch_version()
    finally:
        get_torch_version.cache_clear()

    # then
    assert result == expected_result


def test_get_torchvision_version_when_torch_not_available() -> None:
    # given
    get_torchvision_version.cache_clear()
    try:
        with mock.patch("builtins.__import__") as import_mock:
            import_mock.side_effect = ImportError()

            # when
            result = get_torchvision_version()
    finally:
        get_torchvision_version.cache_clear()

    # then
    assert result is None


def test_get_torchvision_version_when_torch_available_but_version_not_parsable() -> (
    None
):
    # given
    get_torchvision_version.cache_clear()
    try:
        with mock.patch("builtins.__import__") as import_mock:
            torch_mock = MagicMock()
            torch_mock.__version__ = "invalid"
            import_mock.return_value = torch_mock

            # when
            result = get_torchvision_version()
    finally:
        get_torchvision_version.cache_clear()

    # then
    assert result is None


@pytest.mark.parametrize(
    "version, expected_result",
    [
        ("2.6.0", Version("2.6.0")),
        ("2.6.0+cu118", Version("2.6.0")),
        ("2.6.0+cu126", Version("2.6.0")),
        ("2.6.0++gitabcdef1", Version("2.6.0")),
    ],
)
def test_get_torchvision_version_when_torch_available_but_version_parsable(
    version: str, expected_result: Version
) -> None:
    # given
    get_torchvision_version.cache_clear()
    try:
        with mock.patch("builtins.__import__") as import_mock:
            torch_mock = MagicMock()
            torch_mock.__version__ = version
            import_mock.return_value = torch_mock

            # when
            result = get_torchvision_version()
    finally:
        get_torchvision_version.cache_clear()

    # then
    assert result == expected_result


def test_ensure_jetson_l4t_declared_for_jetson_hardware_when_invalid_configuration() -> (
    None
):
    # when
    with pytest.raises(RuntimeIntrospectionError):
        ensure_jetson_l4t_declared_for_jetson_hardware(
            gpu_devices=["some", "orin"],
            l4t_version=None,
        )


def test_ensure_jetson_l4t_declared_for_jetson_hardware_when_valid_configuration() -> (
    None
):
    # when
    ensure_jetson_l4t_declared_for_jetson_hardware(
        gpu_devices=["some"],
        l4t_version=None,
    )
