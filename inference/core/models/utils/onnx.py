from typing import Dict, List, Tuple, Union

_OFFLINE_TRT_OPTIONS_FORCED_DISABLED = {
    "trt_engine_cache_enable",
    "trt_timing_cache_enable",
    "trt_force_timing_cache",
    "trt_dump_subgraphs",
    "trt_dump_ep_context_model",
}
_OFFLINE_TRT_PATH_OPTIONS = {
    "trt_engine_cache_path",
    "trt_engine_cache_prefix",
    "trt_timing_cache_path",
    "trt_ep_context_file_path",
    "trt_onnx_model_folder_path",
}


def has_trt(providers: List[Union[Tuple[str, Dict], str]]) -> bool:
    for p in providers:
        if isinstance(p, tuple):
            name = p[0]
        else:
            name = p
        if name == "TensorrtExecutionProvider":
            return True
    return False


def disable_onnxruntime_trt_file_outputs(
    provider: Union[Tuple[str, Dict], str],
) -> Union[Tuple[str, Dict], str]:
    """Return a copied TensorRT tuple that cannot write runtime cache files."""

    if (
        not isinstance(provider, tuple)
        or len(provider) != 2
        or provider[0] != "TensorrtExecutionProvider"
    ):
        return provider
    provider_options = dict(provider[1])
    for option_name in _OFFLINE_TRT_OPTIONS_FORCED_DISABLED:
        provider_options[option_name] = False
    for option_name in _OFFLINE_TRT_PATH_OPTIONS:
        provider_options.pop(option_name, None)
    return provider[0], provider_options
