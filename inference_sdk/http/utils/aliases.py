# We have a duplicate in inference.models.aliases - please maintain both (to have aliasing work in both libraries)

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
}


def resolve_roboflow_model_alias(model_id: str) -> str:
    return REGISTERED_ALIASES.get(model_id, model_id)
