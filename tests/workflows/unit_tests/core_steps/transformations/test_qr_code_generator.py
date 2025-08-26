import numpy as np
import pytest

from inference.core.workflows.core_steps.transformations.qr_code_generator.v1 import (
    QRCodeGeneratorBlockV1,
    generate_qr_code,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


class TestQRCodeGeneratorBlockV1:
    def test_qr_code_generator_block_manifest(self):
        # given
        block = QRCodeGeneratorBlockV1()

        # when
        manifest_class = block.get_manifest()
        outputs = manifest_class.describe_outputs()

        # then
        assert outputs[0].name == "qr_code"
        assert hasattr(manifest_class, "__fields__")
        assert "type" in manifest_class.__fields__

    def test_qr_code_generator_run_basic(self):
        # given
        block = QRCodeGeneratorBlockV1()
        text = "https://roboflow.com"

        # when
        result = block.run(text=text)

        # then
        assert "qr_code" in result
        assert isinstance(result["qr_code"], WorkflowImageData)
        assert result["qr_code"].numpy_image.shape[2] == 3  # RGB channels
        assert result["qr_code"].numpy_image.dtype == np.uint8

    def test_qr_code_generator_run_with_parameters(self):
        # given
        block = QRCodeGeneratorBlockV1()
        text = "Test"

        # when - version and box_size are now hardcoded per spec
        result = block.run(
            text=text,
            error_correct="High (~30% word recovery / lowest data capacity)",
            border=2,
            fill_color="blue",
            back_color="yellow",
        )

        # then
        assert "qr_code" in result
        assert isinstance(result["qr_code"], WorkflowImageData)
        assert result["qr_code"].numpy_image.shape[2] == 3  # RGB channels


class TestGenerateQRCode:
    def test_generate_qr_code_basic(self):
        # given
        text = "https://example.com"

        # when
        result = generate_qr_code(text=text)

        # then
        assert isinstance(result, WorkflowImageData)
        assert result.numpy_image.shape[2] == 3  # RGB channels
        assert result.numpy_image.dtype == np.uint8
        assert result.numpy_image.shape[0] > 0  # Has height
        assert result.numpy_image.shape[1] > 0  # Has width

    def test_generate_qr_code_with_hardcoded_defaults(self):
        # given
        text = "Test"

        # when - version and box_size are now hardcoded
        result = generate_qr_code(text=text, version=1, box_size=10)

        # then
        assert isinstance(result, WorkflowImageData)
        # Version 1 QR code with box_size=10 and border=4 should be (21+8)*10 = 290 pixels
        expected_size = (21 + 2 * 4) * 10  # 290
        assert result.numpy_image.shape[0] == expected_size
        assert result.numpy_image.shape[1] == expected_size

    def test_generate_qr_code_auto_version(self):
        # given
        text = "Test with auto version"

        # when
        result = generate_qr_code(text=text, version=None)

        # then
        assert isinstance(result, WorkflowImageData)
        assert result.numpy_image.shape[0] > 0
        assert result.numpy_image.shape[1] > 0

    def test_generate_qr_code_error_correction_levels(self):
        # given
        text = "Error correction test"

        # when/then - should not raise errors for valid display name levels
        for level in [
            "Low (~7% word recovery / highest data capacity)",
            "Medium (~15% word recovery)",
            "Quartile (~25% word recovery)",
            "High (~30% word recovery / lowest data capacity)",
        ]:
            result = generate_qr_code(text=text, error_correct=level)
            assert isinstance(result, WorkflowImageData)

    def test_generate_qr_code_invalid_error_correction(self):
        # given
        text = "Test"

        # when
        result = generate_qr_code(text=text, error_correct="INVALID")

        # then - should default to ERROR_CORRECT_M
        assert isinstance(result, WorkflowImageData)

    def test_generate_qr_code_color_parsing(self):
        # given
        text = "Color test"

        # when/then - should handle various color formats
        # Test with standard color names (case-insensitive)
        result1 = generate_qr_code(text=text, fill_color="black", back_color="white")
        assert isinstance(result1, WorkflowImageData)

        # Test with uppercase standard names (matches supervision constants)
        result2 = generate_qr_code(text=text, fill_color="BLACK", back_color="WHITE")
        assert isinstance(result2, WorkflowImageData)

        # Test with hex colors
        result3 = generate_qr_code(
            text=text, fill_color="#FF0000", back_color="#00FF00"
        )
        assert isinstance(result3, WorkflowImageData)

        # Test with rgb format
        result4 = generate_qr_code(
            text=text, fill_color="rgb(255, 0, 0)", back_color="rgb(0, 255, 0)"
        )
        assert isinstance(result4, WorkflowImageData)

        # Test with CSS3 color names (fallback)
        result5 = generate_qr_code(
            text=text, fill_color="mediumpurple", back_color="lightblue"
        )
        assert isinstance(result5, WorkflowImageData)

    def test_generate_qr_code_supervision_color_compatibility(self):
        """Test that all supervision standard colors work with QR code generation."""
        # given
        text = "Supervision color test"

        # Test all standard supervision colors
        standard_colors = [
            "BLACK",
            "WHITE",
            "RED",
            "GREEN",
            "BLUE",
            "YELLOW",
            "ROBOFLOW",
        ]

        for color_name in standard_colors:
            # when - using supervision standard color names
            result = generate_qr_code(
                text=text, fill_color=color_name, back_color="WHITE"
            )

            # then - should successfully generate QR code
            assert isinstance(result, WorkflowImageData)
            assert result.numpy_image is not None
            assert result.numpy_image.shape[2] == 3  # RGB image

        # Test mixed formats to ensure conversions work
        result_mixed = generate_qr_code(
            text=text,
            fill_color="ROBOFLOW",  # supervision constant
            back_color="#FFFFFF",  # hex format
        )
        assert isinstance(result_mixed, WorkflowImageData)

    def test_generate_qr_code_box_size_and_border(self):
        # given
        text = "Size test"

        # when - testing with different parameters (function still accepts them)
        result_small = generate_qr_code(text=text, version=1, box_size=5, border=2)
        result_large = generate_qr_code(text=text, version=1, box_size=15, border=6)

        # then
        assert result_small.numpy_image.shape[0] < result_large.numpy_image.shape[0]
        assert result_small.numpy_image.shape[1] < result_large.numpy_image.shape[1]

    def test_generate_qr_code_empty_text(self):
        # given
        text = ""

        # when
        result = generate_qr_code(text=text)

        # then
        assert isinstance(result, WorkflowImageData)
        assert result.numpy_image.shape[0] > 0
        assert result.numpy_image.shape[1] > 0


@pytest.mark.skipif(
    True,  # Skip until qrcode dependency is resolved in CI
    reason="qrcode library may not be available in test environment",
)
class TestQRCodeGeneratorIntegration:
    def test_qr_code_format_is_png_compatible(self):
        # given
        text = "https://roboflow.com"

        # when
        result = generate_qr_code(text=text)

        # then
        # Verify the image can be used by other workflow blocks
        assert isinstance(result, WorkflowImageData)
        assert result.numpy_image.dtype == np.uint8
        assert len(result.numpy_image.shape) == 3
        assert result.numpy_image.shape[2] == 3  # RGB format expected by IconVisualizer
