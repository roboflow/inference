# We have a duplicate in inference_sdk.http.utils.aliases - please maintain both
# (to have aliasing work in both libraries)

PALIGEMMA_ALIASES = {
    "paligemma-3b-mix-224": "paligemma-pretrains/1",
    "paligemma-3b-mix-448": "paligemma-pretrains/20",
    "paligemma-3b-ft-cococap-224": "paligemma-pretrains/8",
    "paligemma-3b-ft-screen2words-224": "paligemma-pretrains/9",
    "paligemma-3b-ft-vqav2-224": "paligemma-pretrains/10",
    "paligemma-3b-ft-tallyqa-224": "paligemma-pretrains/11",
    "paligemma-3b-ft-docvqa-224": "paligemma-pretrains/12",
    "paligemma-3b-ft-ocrvqa-224": "paligemma-pretrains/13",
    "paligemma-3b-ft-cococap-448": "paligemma-pretrains/14",
    "paligemma-3b-ft-screen2words-448": "paligemma-pretrains/15",
    "paligemma-3b-ft-vqav2-448": "paligemma-pretrains/16",
    "paligemma-3b-ft-tallyqa-448": "paligemma-pretrains/17",
    "paligemma-3b-ft-docvqa-448": "paligemma-pretrains/18",
    "paligemma-3b-ft-ocrvqa-448": "paligemma-pretrains/19",
}
FLORENCE_ALIASES = {
    "florence-2-base": "florence-pretrains/1",
    "florence-2-large": "florence-pretrains/2",
}
YOLOV11_ALIASES = {
    "yolov11n-seg-640": "coco-dataset-vdnr1/19",
    "yolov11s-seg-640": "coco-dataset-vdnr1/20",
    "yolov11m-seg-640": "coco-dataset-vdnr1/21",
    "yolov11l-seg-640": "coco-dataset-vdnr1/22",
    "yolov11x-seg-640": "coco-dataset-vdnr1/23",
    "yolov11n-640": "coco/25",
    "yolov11s-640": "coco/26",
    "yolov11m-640": "coco/27",
    "yolov11l-640": "coco/28",
    "yolov11x-640": "coco/29",
    "yolov11n-1280": "coco/30",
    "yolov11s-1280": "coco/31",
    "yolov11m-1280": "coco/32",
    "yolov11l-1280": "coco/33",
    "yolov11x-1280": "coco/34",
}
YOLOV11_ALIASES = {
    **YOLOV11_ALIASES,
    **{k.replace("yolov11", "yolo11"): v for k, v in YOLOV11_ALIASES.items()},
}
REGISTERED_ALIASES = {
    "yolov8n-640": "coco/3",
    "yolov8n-1280": "coco/9",
    "yolov8s-640": "coco/6",
    "yolov8s-1280": "coco/10",
    "yolov8m-640": "coco/8",
    "yolov8m-1280": "coco/11",
    "yolov8l-640": "coco/7",
    "yolov8l-1280": "coco/12",
    "yolov8x-640": "coco/5",
    "yolov8x-1280": "coco/13",
    "yolo-nas-s-640": "coco/14",
    "yolo-nas-m-640": "coco/15",
    "yolo-nas-l-640": "coco/16",
    "yolov8n-seg-640": "coco-dataset-vdnr1/2",
    "yolov8n-seg-1280": "coco-dataset-vdnr1/7",
    "yolov8s-seg-640": "coco-dataset-vdnr1/4",
    "yolov8s-seg-1280": "coco-dataset-vdnr1/8",
    "yolov8m-seg-640": "coco-dataset-vdnr1/5",
    "yolov8m-seg-1280": "coco-dataset-vdnr1/9",
    "yolov8l-seg-640": "coco-dataset-vdnr1/6",
    "yolov8l-seg-1280": "coco-dataset-vdnr1/10",
    "yolov8x-seg-640": "coco-dataset-vdnr1/3",
    "yolov8x-seg-1280": "coco-dataset-vdnr1/11",
    "yolov8n-pose-640": "coco-pose-detection/1",
    "yolov8s-pose-640": "coco-pose-detection/2",
    "yolov8m-pose-640": "coco-pose-detection/3",
    "yolov8l-pose-640": "coco-pose-detection/4",
    "yolov8x-pose-640": "coco-pose-detection/5",
    "yolov8x-pose-1280": "coco-pose-detection/6",
    "yolov10n-640": "coco/19",
    "yolov10s-640": "coco/20",
    "yolov10m-640": "coco/21",
    "yolov10b-640": "coco/22",
    "yolov10l-640": "coco/23",
    "yolov10x-640": "coco/24",
    **PALIGEMMA_ALIASES,
    **FLORENCE_ALIASES,
    **YOLOV11_ALIASES,
}


def resolve_roboflow_model_alias(model_id: str) -> str:
    return REGISTERED_ALIASES.get(model_id, model_id)
