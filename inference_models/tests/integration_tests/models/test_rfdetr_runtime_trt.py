from pathlib import Path

import numpy as np
import pytest
import torch

MODEL_ID = "rfdetr-nano"
ORIN_PACKAGE_ID = "bbf73c3014b86386434c8051f132cdf2"


@pytest.mark.slow
@pytest.mark.trt_extras
def test_runtime_negotiated_rfdetr_trt_inference(tmp_path: Path) -> None:
    from inference_models.developer_tools import (
        download_files_to_directory,
        get_model_from_provider,
    )
    from inference_models.models.auto_loaders.auto_negotiation import (
        filter_model_packages_matching_runtime_environment,
    )
    from inference_models.models.rfdetr.optimization.execution_plan import (
        RFDetrExecutionPlan,
    )
    from inference_models.models.rfdetr.optimization.ids import (
        RFDETR_PREPROCESSOR_BASE,
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    )
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )
    from inference_models.weights_providers.entities import FileDownloadSpecs

    device = torch.device("cuda:0")
    model_metadata = get_model_from_provider(model_id=MODEL_ID)
    selected_packages = [
        package
        for package in model_metadata.model_packages
        if package.package_id == ORIN_PACKAGE_ID
    ]
    assert len(selected_packages) == 1

    compatible_packages, discarded_packages = (
        filter_model_packages_matching_runtime_environment(
            model_packages=selected_packages,
            device=device,
            onnx_execution_providers=None,
            trt_engine_host_code_allowed=True,
        )
    )
    assert compatible_packages == selected_packages, [
        discarded.reason for discarded in discarded_packages
    ]
    package = compatible_packages[0]

    file_specs = []
    for artefact in package.package_artefacts:
        assert isinstance(artefact, FileDownloadSpecs)
        file_specs.append(
            (artefact.file_handle, artefact.download_url, artefact.md5_hash)
        )
    download_files_to_directory(
        target_dir=str(tmp_path),
        files_specs=file_specs,
        verify_hash_while_download=True,
        download_files_without_hash=False,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=str(tmp_path),
        device=device,
        engine_host_code_allowed=True,
        rfdetr_execution_plan=RFDetrExecutionPlan(
            preprocessor_id=RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
        ),
    )
    assert (
        model.preprocessor_implementation_id
        == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    )

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    predictions = model(image)

    assert len(predictions) == 1
    assert predictions[0].xyxy.ndim == 2
    assert predictions[0].xyxy.shape[1] == 4

    runtime_metadata = model.optimization_runtime_metadata
    preprocessor_execution = runtime_metadata["last_execution"]["preprocessor"]
    assert (
        preprocessor_execution["requested_id"]
        == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    )
    assert preprocessor_execution["effective_id"] in {
        RFDETR_PREPROCESSOR_BASE,
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    }
    print(
        f"RF-DETR package: model_id={MODEL_ID} package_id={package.package_id} "
        f"trusted_source={package.trusted_source}"
    )
    print(f"RF-DETR optimization runtime metadata: {runtime_metadata}")
