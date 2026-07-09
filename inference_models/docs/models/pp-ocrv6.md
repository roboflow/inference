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

## Performance

End-to-end latency (pre-processing + inference + post-processing), mean over 50 runs after warmup, on **NVIDIA L4** (24 GB, ONNX Runtime `CUDAExecutionProvider`) and an **Apple Silicon MacBook** (arm64, `CPUExecutionProvider`):

| Model | Input | L4 (CUDA) | MacBook (CPU) |
|---|---|---|---|
| `pp-ocrv6-det` tiny   | 640×480 image           | 19.5 ms | 63.4 ms  |
| `pp-ocrv6-det` small  | 640×480 image           | 23.0 ms | 103.9 ms |
| `pp-ocrv6-det` medium | 640×480 image           | 44.0 ms | 445.8 ms |
| `pp-ocrv6-rec` tiny   | batch of 8 line crops   | 3.7 ms  | 17.6 ms  |
| `pp-ocrv6-rec` small  | batch of 8 line crops   | 26.7 ms | 96.7 ms  |
| `pp-ocrv6-rec` medium | batch of 8 line crops   | 33.0 ms | 417.5 ms |

Predictions match the original PaddlePaddle implementation of the same weights — verified on the L4 against native Paddle inference (`paddlex`), with identical detections and exact-string recognition across all three variants. On L4 the GPU forward is the minor cost; end-to-end latency is dominated by CPU-side pre-processing (detection image resize/normalize) and CTC decoding (recognition, scaling with the ~18.7k-character vocabulary of the `small`/`medium` variants).

## License

**Apache 2.0**

!!! info "Open Source License"
    PP-OCRv6 code and weights are released by PaddlePaddle under Apache 2.0, making them free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Usage

Both detection (`pp-ocrv6-det`) and recognition (`pp-ocrv6-rec`) are registered in `tiny` / `small` / `medium` variants and load via `AutoModel.from_pretrained`.

### Input contract

Both models accept `np.ndarray` (`bgr` assumed), `torch.Tensor` (`rgb` assumed, `CHW` / `BCHW`), or lists of either. Integer images are read as `[0, 255]`; floating-point images are assumed to already be on the `[0, 255]` scale. This matches the input-scale convention of the other ONNX models in this package.

### Pipeline class (recommended)

`PPOCRv6Pipeline` bundles detection, perspective-cropping, reading-order grouping, and recognition behind a single call. It takes model ids directly and returns one `PPOCRv6PipelineResult` per input image, with `text` (all lines joined in reading order), `line_texts` (one per kept detection), and the reordered `detections` (`None` when the detection stage is disabled).

```python
import cv2

from inference_models.models.pp_ocrv6.pp_ocrv6_pipeline import PPOCRv6Pipeline

pipeline = PPOCRv6Pipeline.from_pretrained(
    "pp-ocrv6-det/small", "pp-ocrv6-rec/small"
)

image = cv2.imread("document.png")
result = pipeline(image)[0]
print(result.text)
```

`from_pretrained` forwards any extra keyword arguments (e.g. `onnx_execution_providers`) to both underlying models. To reuse already-loaded models, construct the pipeline directly with `PPOCRv6Pipeline(det_model=detector, rec_model=recognizer)`.

### Two-stage composition

For full control over cropping and ordering, load the two models with `AutoModel` and compose them yourself:

```python
import cv2

from inference_models import AutoModel

det = AutoModel.from_pretrained("pp-ocrv6-det/small")
rec = AutoModel.from_pretrained("pp-ocrv6-rec/small")

image = cv2.imread("document.png")

detections = det(image)[0]
crops = [
    image[int(y1):int(y2), int(x1):int(x2)]
    for x1, y1, x2, y2 in sorted(detections.xyxy.tolist(), key=lambda b: (b[1], b[0]))
]
texts = rec(crops)
print("\n".join(texts))
```

`from_pretrained` also accepts a local package directory containing `inference.onnx` and `inference.yml`.

Either stage is optional (passing only one model, or `None` for the other path in `from_pretrained`, skips it; passing neither raises `ValueError`):

- **Detect-only** (recognition skipped) — construct with `rec_model=None`. Each result carries reading-order `detections` with empty `line_texts` and `text=""`; recognition is never invoked.
- **Recognize-only** (detection skipped) — construct with `det_model=None`. Each input image is treated as a single text-line crop and passed straight to recognition; each result has `text` set to the recognized string, `line_texts=[text]`, and empty `detections`.
