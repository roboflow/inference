"""Convert MediaPipe's hand landmarker bundle into inference-models ONNX packages.

``hand_landmarker.task`` is a zip of two TFLite graphs: the palm detector and
the hand landmarks model (Apache 2.0, see the MediaPipe model card). This
script converts each with tf2onnx, makes the batch dimension dynamic, and
writes three model packages:

* ``mediapipe-palm-detector``: the palm detector graph,
* ``mediapipe-hand-landmarks``: the landmarks graph,
* ``mediapipe-hands``: the composite, which carries only its config and names
  the other two as its ``palmDetector`` and ``handLandmarks`` dependencies.

The conversion needs ``tensorflow``, ``tf2onnx`` and ``onnx``, none of which
inference-models depends on at runtime:

    pip install tensorflow tf2onnx onnx onnxruntime
    python development/converters/convert_mediapipe_hands_tflite_to_onnx.py \
        --output-dir /tmp/mediapipe-hands

Pass ``--task-file`` to convert a bundle that is already on disk; it must match
the pinned SHA-256 unless ``--allow-unpinned`` is given.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper

TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
TASK_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
PALM_DETECTOR_TFLITE = "hand_detector.tflite"
HAND_LANDMARKS_TFLITE = "hand_landmarks_detector.tflite"
OPSET = 17

PALM_DETECTOR_PACKAGE = "mediapipe-palm-detector"
HAND_LANDMARKS_PACKAGE = "mediapipe-hand-landmarks"
HANDS_PACKAGE = "mediapipe-hands"
BATCH_DIMENSION = "batch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work_dir:
        task_path = args.task_file or os.path.join(work_dir, "hand_landmarker.task")
        if args.task_file is None:
            urllib.request.urlretrieve(TASK_URL, task_path)
        digest = _sha256(task_path)
        if digest != TASK_SHA256 and not args.allow_unpinned:
            sys.exit(
                f"{task_path} has SHA-256 {digest}, expected {TASK_SHA256}. "
                "Pass --allow-unpinned to convert a different bundle."
            )
        with zipfile.ZipFile(task_path) as bundle:
            bundle.extractall(
                work_dir, members=[PALM_DETECTOR_TFLITE, HAND_LANDMARKS_TFLITE]
            )

        palm_dir = os.path.join(args.output_dir, PALM_DETECTOR_PACKAGE)
        landmarks_dir = os.path.join(args.output_dir, HAND_LANDMARKS_PACKAGE)
        hands_dir = os.path.join(args.output_dir, HANDS_PACKAGE)
        for package_dir in (palm_dir, landmarks_dir, hands_dir):
            os.makedirs(package_dir, exist_ok=True)

        _convert(
            tflite_path=os.path.join(work_dir, PALM_DETECTOR_TFLITE),
            onnx_path=os.path.join(palm_dir, "weights.onnx"),
        )
        _convert(
            tflite_path=os.path.join(work_dir, HAND_LANDMARKS_TFLITE),
            onnx_path=os.path.join(landmarks_dir, "weights.onnx"),
        )
        _write_json(
            os.path.join(palm_dir, "model_config.json"),
            {
                "model_architecture": PALM_DETECTOR_PACKAGE,
                "task_type": "keypoint-detection",
                "backend_type": "onnx",
            },
        )
        _write_json(
            os.path.join(landmarks_dir, "model_config.json"),
            {
                "model_architecture": HAND_LANDMARKS_PACKAGE,
                "task_type": "keypoint-detection",
                "backend_type": "onnx",
            },
        )
        _write_json(
            os.path.join(hands_dir, "model_config.json"),
            {
                "model_architecture": HANDS_PACKAGE,
                "task_type": "keypoint-detection",
                "backend_type": "onnx",
            },
        )
        _write_json(
            os.path.join(hands_dir, "mediapipe_hands_config.json"),
            {"source_task_sha256": digest, "source_task_url": TASK_URL},
        )
    print(
        f"Wrote {PALM_DETECTOR_PACKAGE}, {HAND_LANDMARKS_PACKAGE} and {HANDS_PACKAGE} to {args.output_dir}"
    )


def _convert(tflite_path: str, onnx_path: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--tflite",
            tflite_path,
            "--output",
            onnx_path,
            "--opset",
            str(OPSET),
        ],
        check=True,
    )
    model = onnx.load(onnx_path)
    _make_batch_dynamic(model)
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


def _make_batch_dynamic(model: onnx.ModelProto) -> None:
    """Free the batch dimension tf2onnx pins to 1.

    The graph inputs and outputs get a symbolic first dimension, and the
    palm detector's output reshapes, which tf2onnx folds to ``[1, -1, n]``,
    copy the batch from their input instead (``0``).
    """
    graph = model.graph
    for value in list(graph.input) + list(graph.output):
        first = value.type.tensor_type.shape.dim[0]
        first.ClearField("dim_value")
        first.dim_param = BATCH_DIMENSION
    del graph.value_info[:]
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    for node in graph.node:
        if node.op_type != "Reshape" or node.input[1] not in initializers:
            continue
        shape = numpy_helper.to_array(initializers[node.input[1]]).copy()
        if shape.size and shape[0] == 1:
            shape[0] = 0
            initializers[node.input[1]].CopyFrom(
                numpy_helper.from_array(shape.astype(np.int64), node.input[1])
            )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str, content: dict) -> None:
    with open(path, "w") as file:
        json.dump(content, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
