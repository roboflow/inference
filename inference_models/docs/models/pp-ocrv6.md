# PP-OCRv6 - Text Detection & Recognition

PP-OCRv6 is the latest generation of PaddlePaddle's ultra-lightweight OCR system. It ships as two independent models — a DBNet-based **text detector** and a CTC-based **text recognizer** — that chain into a complete two-stage OCR pipeline.

## Overview

**Resources**: [PaddleOCR GitHub Repository](https://github.com/PaddlePaddle/PaddleOCR), [PP-OCRv6 models on Hugging Face](https://huggingface.co/PaddlePaddle)

Key features:

- **Ultra-lightweight** - `tiny` / `small` / `medium` variants, from a few MB up
- **Two-stage pipeline** - Independent detection and recognition stages
- **Dynamic input shapes** - Detector accepts arbitrary image sizes; recognizer adapts its width to the text-line aspect ratio
- **ONNX backend** - Runs through ONNX Runtime with IO binding on CUDA devices

## Models

### Text Detection (`pp-ocrv6-det`)

**Task**: `object-detection`. Detects text regions with a DBNet probability map. Returns axis-aligned bounding boxes; the tight four-point quadrilateral of each region is preserved in `Detections.bboxes_metadata["polygon"]` so downstream recognition can crop rotated text lines accurately.

### Text Recognition (`pp-ocrv6-rec`)

**Task**: `text-only-ocr`. Reads text from cropped single text-line images and returns one string per crop. Inputs should be crops produced by a text detector — the model does not localize text by itself.

### When to Use PP-OCRv6

- ✅ **Printed text** - Documents, labels, signs, rendered text
- ✅ **Resource-constrained deployments** - Smallest variants run comfortably on CPU
- ✅ **Custom OCR pipelines** - Detection and recognition compose freely with your own cropping / ordering logic

### When to Use Other OCR Models

- **DocTR**: Better for structured documents (invoices, forms, scanned pages)
- **EasyOCR**: Broader multi-language scene-text support out of the box
- **TrOCR**: Transformer-based recognition of pre-cropped text lines

## License

**Apache 2.0**

!!! info "Open Source License"
    PP-OCRv6 code and weights are released by PaddlePaddle under Apache 2.0, making them free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Usage

!!! warning "Model registration pending"
    PP-OCRv6 packages are not yet registered in the Roboflow model registry, so
    `AutoModel.from_pretrained("pp-ocrv6-det/<variant>")` does not resolve yet.
    Until registration lands, load the models from a local package directory
    containing `inference.onnx` and `inference.yml` (as exported to the
    [PaddlePaddle `*_onnx` repositories on Hugging Face](https://huggingface.co/PaddlePaddle)).

### Input contract

Both models accept `np.ndarray` (`bgr` assumed), `torch.Tensor` (`rgb` assumed, `CHW` / `BCHW`), or lists of either. Integer images are read as `[0, 255]`; floating-point images are read as `[0, 1]` and rescaled (values above `1.0` are treated as already being on the `[0, 255]` scale).

### Two-stage pipeline

```python
import cv2

from inference_models.models.pp_ocrv6.pp_ocrv6_detection_onnx import PPOCRv6DetectionOnnx
from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_onnx import PPOCRv6RecognitionOnnx

detector = PPOCRv6DetectionOnnx.from_pretrained("<path-to-det-package>")
recognizer = PPOCRv6RecognitionOnnx.from_pretrained("<path-to-rec-package>")

image = cv2.imread("document.png")

detections = detector(image)[0]
crops = [
    image[int(y1):int(y2), int(x1):int(x2)]
    for x1, y1, x2, y2 in sorted(detections.xyxy.tolist(), key=lambda b: (b[1], b[0]))
]
texts = recognizer(crops)
print("\n".join(texts))
```

See [`examples/pp_ocrv6/two_stage_ocr.py`](https://github.com/roboflow/inference/blob/main/inference_models/examples/pp_ocrv6/two_stage_ocr.py) for a complete, self-contained runnable example (including perspective cropping via the detection polygons and reading-order grouping).
