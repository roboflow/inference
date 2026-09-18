from inference.core import env


def validate_model_selection(request):
    selectors = model_selection_kwargs(request)
    if "model_package_id" in selectors and (
        "backend" in selectors or "quantization" in selectors
    ):
        raise ValueError(
            "model_package_id cannot be combined with backend or quantization."
        )
    if selectors and not env.USE_INFERENCE_MODELS:
        raise ValueError("Model package selection requires USE_INFERENCE_MODELS=true.")
    return request


def model_selection_kwargs(request) -> dict:
    return {
        name: value
        for name in ("model_package_id", "backend", "quantization")
        if (value := getattr(request, name, None)) is not None
    }
