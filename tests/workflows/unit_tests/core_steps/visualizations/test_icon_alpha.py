import numpy as np
import pytest
import supervision as sv
import base64
import cv2
from inference.core.workflows.core_steps.visualizations.icon.v1 import (
    IconVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_icon_with_alpha_from_workflow_input_numpy():
    """Test that alpha channel is preserved when icon comes from workflow input as numpy."""
    # given
    block = IconVisualizationBlockV1()
    
    # Create a test background image (white)
    bg_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    test_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="background"),
        numpy_image=bg_image,
    )
    
    # Create an icon with alpha channel (red circle with transparent background)
    icon_with_alpha = np.zeros((100, 100, 4), dtype=np.uint8)
    center = (50, 50)
    for y in range(100):
        for x in range(100):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist <= 40:
                # Red circle with full opacity
                icon_with_alpha[y, x] = [0, 0, 255, 255]  # BGRA
            else:
                # Transparent background
                icon_with_alpha[y, x] = [0, 0, 0, 0]
    
    # Create WorkflowImageData from numpy array (simulating workflow input)
    test_icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon_input"),
        numpy_image=icon_with_alpha,  # Has 4 channels with alpha
    )
    
    # Run the block in static mode
    output = block.run(
        image=test_image,
        copy_image=True,
        mode="static",
        icon=test_icon,
        predictions=None,
        icon_width=100,
        icon_height=100,
        position=None,
        x_position=200,
        y_position=200,
    )
    
    result_image = output['image'].numpy_image
    
    # Check that transparency was preserved
    # The corners of where the icon was placed should still be white
    corner_positions = [
        (200, 200),  # Top-left corner of icon placement
        (299, 200),  # Top-right corner
        (200, 299),  # Bottom-left corner
        (299, 299),  # Bottom-right corner
    ]
    
    for x, y in corner_positions:
        pixel_color = result_image[y, x]
        # Should be white or very close to white
        assert np.all(pixel_color > 250), \
            f"Expected white background at ({x},{y}) due to transparency, got {pixel_color}"
    
    # Check that the red circle is visible in the center
    center_x, center_y = 250, 250
    center_color = result_image[center_y, center_x]
    
    # Should be red (high red channel, low blue/green)
    assert center_color[2] > 200, f"Red channel too low at center: {center_color}"
    assert center_color[0] < 100 and center_color[1] < 100, \
        f"Blue/Green should be low at center: {center_color}"


def test_icon_with_alpha_from_base64_input():
    """Test that alpha channel is preserved when icon comes as base64 (API input scenario)."""
    # given
    block = IconVisualizationBlockV1()
    
    # Create a test background image (white)
    bg_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    test_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="background"),
        numpy_image=bg_image,
    )
    
    # Create an icon with alpha channel
    icon_with_alpha = np.zeros((100, 100, 4), dtype=np.uint8)
    center = (50, 50)
    for y in range(100):
        for x in range(100):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist <= 40:
                # Blue circle with full opacity
                icon_with_alpha[y, x] = [255, 0, 0, 255]  # BGRA
            else:
                # Transparent background
                icon_with_alpha[y, x] = [0, 0, 0, 0]
    
    # Encode as PNG base64 (preserves alpha)
    _, png_buffer = cv2.imencode('.png', icon_with_alpha)
    png_base64 = base64.b64encode(png_buffer).decode('ascii')
    
    # Create WorkflowImageData from base64 (simulating API input)
    test_icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon_base64"),
        base64_image=png_base64,
    )
    
    # Run the block in static mode
    output = block.run(
        image=test_image,
        copy_image=True,
        mode="static",
        icon=test_icon,
        predictions=None,
        icon_width=100,
        icon_height=100,
        position=None,
        x_position=200,
        y_position=200,
    )
    
    result_image = output['image'].numpy_image
    
    # Check that transparency was preserved
    corner_positions = [
        (200, 200),  # Top-left corner
        (299, 200),  # Top-right corner  
        (200, 299),  # Bottom-left corner
        (299, 299),  # Bottom-right corner
    ]
    
    for x, y in corner_positions:
        pixel_color = result_image[y, x]
        # Should be white or very close to white (allowing small variations)
        assert np.all(pixel_color > 250), \
            f"Expected white background at ({x},{y}) due to transparency, got {pixel_color}"
    
    # Check that the blue circle is visible in the center
    center_x, center_y = 250, 250
    center_color = result_image[center_y, center_x]
    
    # Should be blue (high blue channel, low red/green)
    assert center_color[0] > 200, f"Blue channel too low at center: {center_color}"
    assert center_color[1] < 100 and center_color[2] < 100, \
        f"Red/Green should be low at center: {center_color}"
