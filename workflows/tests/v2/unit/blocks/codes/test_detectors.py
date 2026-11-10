"""Barcode and QR code detectors on generated codes.

Every fixture is drawn in the test with ``qrcode`` or zxing-cpp's writer, so
the expected text and position are known exactly. The writers differ between
zxing-cpp 2.2 and 2.3 in output size, so barcode assertions use the placed
region, never a fixed pixel size.
"""

from typing import List, Tuple

import cv2
import numpy as np
import pytest
import qrcode
import torch
import zxingcpp
from roboflow_workflows.v2.blocks.codes.barcode_detector.v1 import BarcodeDetectorBlock
from roboflow_workflows.v2.blocks.codes.qr_code_detector.v1 import QRCodeDetectorBlock
from roboflow_workflows.v2.blocks.codes._shared.detectors import code_detections, host_luminance
from roboflow_workflows.v2.blocks.codes.qr_code_generator.v1 import QRCodeGeneratorBlock
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import (
    BAR_CODE_DETECTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record

MODULE = 6
QUIET = 4
TOLERANCE = 3.0


def _qr(text: str) -> np.ndarray:
    code = qrcode.QRCode(box_size=MODULE, border=QUIET)
    code.add_data(text)
    code.make(fit=True)
    gray = np.asarray(code.make_image().convert("L"), dtype=np.uint8)

    return gray


def _barcode(text: str, code_format: "zxingcpp.BarcodeFormat") -> np.ndarray:
    drawn = np.asarray(
        zxingcpp.write_barcode(code_format, text, width=300, height=80, quiet_zone=20)
    )
    # Thicker bars keep the fixture readable after placement.
    gray = np.repeat(np.repeat(drawn, 2, axis=0), 2, axis=1)

    return gray


def _canvas(height: int, width: int, *codes: Tuple[np.ndarray, int, int]) -> np.ndarray:
    canvas = np.full((height, width), 255, dtype=np.uint8)
    for code, x, y in codes:
        canvas[y : y + code.shape[0], x : x + code.shape[1]] = code

    return canvas


def _rgb(gray: np.ndarray) -> ImageData:
    image = ImageData.from_numpy_rgb(np.stack([gray] * 3, axis=-1))

    return image


def _qr_box(code: np.ndarray, x: int, y: int) -> List[float]:
    # The decoders report the code without its quiet zone.
    margin = QUIET * MODULE
    box = [
        x + margin,
        y + margin,
        x + code.shape[1] - margin,
        y + code.shape[0] - margin,
    ]

    return [float(value) for value in box]


def _texts(predictions) -> List[str]:
    texts = [row["data"] for row in predictions.bboxes_metadata]

    return texts


def test_qr_round_trip_reads_text_and_box_with_v1_metadata() -> None:
    # given
    code = _qr("https://roboflow.com")
    image = _rgb(_canvas(300, 400, (code, 60, 40)))

    # when
    predictions = QRCodeDetectorBlock().run(image=image)["predictions"]

    # then
    QR_CODE_DETECTION_KIND.check(predictions)
    assert _texts(predictions) == ["https://roboflow.com"]
    assert predictions.xyxy.dtype == torch.float32
    assert torch.allclose(
        predictions.xyxy[0], torch.tensor(_qr_box(code, 60, 40)), atol=TOLERANCE
    )
    assert predictions.class_id.tolist() == [0]
    assert predictions.confidence.tolist() == [1.0]
    assert predictions.image_metadata["class_names"] == {0: "qr_code"}
    assert predictions.image_metadata["prediction_type"] == "qrcode-detection"
    assert frame_of(predictions).image_id == image.image_id
    assert row_record(predictions).rows == 1


@pytest.mark.parametrize(
    "block, code_kind",
    [
        (BarcodeDetectorBlock, BAR_CODE_DETECTION_KIND),
        (QRCodeDetectorBlock, QR_CODE_DETECTION_KIND),
    ],
)
def test_predictions_have_their_code_kind_and_the_object_detection_kind(
    block, code_kind
) -> None:
    # given
    image = _rgb(_canvas(200, 200, (_qr("kinds"), 20, 20)))

    # when
    predictions = block().run(image=image)["predictions"]

    # then: the payload is plain Detections, so box consumers can read it.
    assert block.outputs["predictions"].kinds == (
        code_kind,
        OBJECT_DETECTION_PREDICTION_KIND,
    )
    code_kind.check(predictions)
    OBJECT_DETECTION_PREDICTION_KIND.check(predictions)


@pytest.mark.parametrize("text", ["https://roboflow.com", "LOT-42"])
def test_qr_detector_reads_the_generators_default_layout(text) -> None:
    # given: border 4 and 10 px modules; V1's OpenCV decoder misses the URL.
    image = QRCodeGeneratorBlock().run(
        text=text,
        error_correct="Medium (~15% word recovery)",
        border=4,
        fill_color="BLACK",
        back_color="WHITE",
    )["qr_code"]

    # when
    predictions = QRCodeDetectorBlock().run(image=image)["predictions"]

    # then
    assert _texts(predictions) == [text]


def test_qr_detector_ignores_linear_barcodes() -> None:
    # given
    code_128 = _barcode("CODE-128-XY", zxingcpp.BarcodeFormat.Code128)
    image = _rgb(_canvas(300, 900, (code_128, 20, 20), (_qr("only-qr"), 700, 60)))

    # when
    predictions = QRCodeDetectorBlock().run(image=image)["predictions"]

    # then
    assert _texts(predictions) == ["only-qr"]


def test_barcode_detector_reads_linear_formats_and_qr_codes() -> None:
    # given
    code_128 = _barcode("CODE-128-XY", zxingcpp.BarcodeFormat.Code128)
    ean_13 = _barcode("5901234123457", zxingcpp.BarcodeFormat.EAN13)
    qr_code = _qr("qr-too")
    gray = _canvas(700, 900, (code_128, 20, 20), (ean_13, 20, 300), (qr_code, 650, 450))
    image = _rgb(gray)

    # when
    predictions = BarcodeDetectorBlock().run(image=image)["predictions"]

    # then
    BAR_CODE_DETECTION_KIND.check(predictions)
    assert sorted(_texts(predictions)) == ["5901234123457", "CODE-128-XY", "qr-too"]
    assert predictions.image_metadata["class_names"] == {0: "barcode"}
    assert predictions.image_metadata["prediction_type"] == "barcode-detection"
    by_text = dict(zip(_texts(predictions), predictions.xyxy.tolist()))
    x0, y0, x1, y1 = by_text["CODE-128-XY"]
    assert 20 <= x0 < x1 <= 20 + code_128.shape[1]
    assert 20 <= y0 < y1 <= 20 + code_128.shape[0]
    assert by_text["qr-too"] == pytest.approx(_qr_box(qr_code, 650, 450), abs=TOLERANCE)


@pytest.mark.parametrize("block", [BarcodeDetectorBlock, QRCodeDetectorBlock])
def test_several_qr_codes_are_separate_aligned_rows(block) -> None:
    # given
    first, second = _qr("first"), _qr("second")
    image = _rgb(_canvas(400, 700, (first, 10, 10), (second, 400, 180)))

    # when
    predictions = block().run(image=image)["predictions"]

    # then
    boxes = dict(zip(_texts(predictions), predictions.xyxy.tolist()))
    assert sorted(boxes) == ["first", "second"]
    assert boxes["first"] == pytest.approx(_qr_box(first, 10, 10), abs=TOLERANCE)
    assert boxes["second"] == pytest.approx(_qr_box(second, 400, 180), abs=TOLERANCE)
    assert row_record(predictions).rows == 2


@pytest.mark.parametrize("angle", [15.0, 30.0, 45.0, 90.0])
@pytest.mark.parametrize("block", [BarcodeDetectorBlock, QRCodeDetectorBlock])
def test_rotated_code_box_encloses_all_four_corners(block, angle) -> None:
    # given
    code = _qr("rotated")
    gray = _canvas(400, 400, (code, 130, 130))
    rotation = cv2.getRotationMatrix2D((200.0, 200.0), angle, 1.0)
    rotated = cv2.warpAffine(gray, rotation, (400, 400), borderValue=255)
    x0, y0, x1, y1 = _qr_box(code, 130, 130)
    square = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]])
    corners = square @ rotation.T
    expected = [*corners.min(axis=0), *corners.max(axis=0)]

    # when
    predictions = block().run(image=_rgb(rotated))["predictions"]

    # then
    assert _texts(predictions) == ["rotated"]
    # At 30 degrees V1's two-corner box would be about 60 px narrower in y.
    assert predictions.xyxy[0].tolist() == pytest.approx(expected, abs=2 * TOLERANCE)


@pytest.mark.parametrize("block", [BarcodeDetectorBlock, QRCodeDetectorBlock])
@pytest.mark.parametrize("size_hw", [(1, 1), (5, 3), (120, 80)])
def test_image_without_codes_gives_empty_predictions_with_frame(block, size_hw) -> None:
    # given
    image = _rgb(np.full(size_hw, 255, dtype=np.uint8))

    # when
    predictions = block().run(image=image)["predictions"]

    # then
    block.outputs["predictions"].kinds[0].check(predictions)
    assert predictions.xyxy.shape == (0, 4)
    assert predictions.xyxy.dtype == torch.float32
    assert predictions.class_id.shape == (0,)
    assert predictions.bboxes_metadata == []
    assert frame_of(predictions).size_hw == size_hw
    assert row_record(predictions).rows == 0


@pytest.mark.parametrize("block", [BarcodeDetectorBlock, QRCodeDetectorBlock])
def test_grayscale_and_noncontiguous_inputs_read_the_same_code(block) -> None:
    # given
    code = _qr("layouts")
    gray = _canvas(260, 260, (code, 40, 40))
    single_channel = ImageData.from_tensor(torch.from_numpy(gray)[None])
    rgb = torch.from_numpy(np.stack([gray] * 3))
    wide = torch.zeros((3, 260, 520), dtype=torch.uint8)
    wide[:, :, ::2] = rgb
    strided = ImageData.from_tensor(wide[:, :, ::2])
    assert not strided.tensor_image.is_contiguous()

    # when
    results = [
        block().run(image=image)["predictions"] for image in (single_channel, strided)
    ]

    # then
    for predictions in results:
        assert _texts(predictions) == ["layouts"]
        assert predictions.xyxy[0].tolist() == pytest.approx(
            _qr_box(code, 40, 40), abs=TOLERANCE
        )


def test_qr_codes_on_a_crop_keep_crop_coordinates_and_root_mapping() -> None:
    # given
    code = _qr("cropped")
    parent = _rgb(_canvas(400, 500, (code, 250, 200)))
    crop = parent.crop([200, 150, 480, 390])

    # when
    predictions = QRCodeDetectorBlock().run(image=crop)["predictions"]

    # then
    x0, y0, _, _ = predictions.xyxy[0].tolist()
    expected = _qr_box(code, 250 - 200, 200 - 150)
    assert (x0, y0) == pytest.approx(expected[:2], abs=TOLERANCE)
    frame = frame_of(predictions)
    assert frame.image_id == crop.image_id
    assert frame.root.map_xy(x0, y0) == pytest.approx((x0 + 200, y0 + 150))


def test_luminance_matches_opencv_and_crosses_to_host_once() -> None:
    # given
    generator = torch.Generator().manual_seed(7)
    rgb = torch.randint(0, 256, (3, 31, 17), dtype=torch.uint8, generator=generator)
    gray = torch.randint(0, 256, (1, 9, 5), dtype=torch.uint8, generator=generator)
    hwc = np.ascontiguousarray(rgb.permute(1, 2, 0).numpy())

    # when
    from_rgb = host_luminance(ImageData.from_tensor(rgb))
    from_gray = host_luminance(ImageData.from_tensor(gray))

    # then
    assert np.array_equal(from_rgb, cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY))
    assert from_rgb.flags["C_CONTIGUOUS"] and from_rgb.dtype == np.uint8
    assert np.shares_memory(from_gray, gray.numpy())


def test_boxes_are_clipped_to_the_image_and_texts_stay_aligned() -> None:
    # given
    image = _rgb(np.zeros((50, 80), dtype=np.uint8))
    corners = [
        [(-3.5, -2.0), (40.0, -2.0), (40.0, 30.0), (-3.5, 30.0)],
        [(60.0, 20.0), (90.0, 25.0), (85.0, 55.0), (55.0, 50.0)],
    ]

    # when
    predictions = code_detections(
        image,
        corners=corners,
        texts=["edge", ""],
        class_name="qr_code",
        prediction_type="qrcode-detection",
    )

    # then
    assert predictions.xyxy.tolist() == [
        [0.0, 0.0, 40.0, 30.0],
        [55.0, 20.0, 80.0, 50.0],
    ]
    assert _texts(predictions) == ["edge", ""]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("block", [BarcodeDetectorBlock, QRCodeDetectorBlock])
def test_cuda_image_gives_cuda_predictions(block) -> None:
    # given
    image = ImageData.from_tensor(_rgb(_qr("cuda")).tensor_image.cuda())

    # when
    predictions = block().run(image=image)["predictions"]

    # then
    assert _texts(predictions) == ["cuda"]
    assert predictions.xyxy.device == image.device
    assert predictions.class_id.device == image.device
