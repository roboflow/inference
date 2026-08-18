import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_runtime_negotiated_rfdetr_trt_inference() -> None:
    from inference_models import AutoModel, BackendType
    from inference_models.models.rfdetr.optimization.ids import (
        RFDETR_PREPROCESSOR_BASE,
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    )
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-nano",
        backend=BackendType.TRT,
        device=torch.device("cuda:0"),
        use_auto_resolution_cache=False,
    )
    assert isinstance(model, RFDetrForObjectDetectionTRT)
    assert (
        model.preprocessor_implementation_id
        == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    )

    image = np.zeros((640, 640, 3), dtype=np.uint8)
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
    print(f"RF-DETR optimization runtime metadata: {runtime_metadata}")
