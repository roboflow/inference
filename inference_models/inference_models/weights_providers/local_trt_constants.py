LOCAL_TRT_PACKAGE_PREFIX = "localtrt"
LOCAL_TRT_MANIFEST_FILE = "local_trt_package_manifest.json"

ALLOWED_LOCAL_TRT_FILE_HANDLES = frozenset(
    {
        "inference_config.json",
        "class_names.txt",
        "trt_config.json",
        "engine.plan",
        "keypoints_metadata.json",
    }
)
