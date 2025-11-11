#!/usr/bin/env python3
"""Generate a small test video for integration tests."""

import cv2
import numpy as np
from pathlib import Path

def generate_test_video():
    """Generate a small test video with 10 colored frames."""
    output_path = Path(__file__).parent / "test_video.mp4"

    width, height = 640, 480
    fps = 30
    num_frames = 10

    # Use H264 codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Generate frames with different colors
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 128, 128),# Gray
        (255, 255, 255),# White
        (0, 0, 0),      # Black
        (128, 64, 32),  # Brown
    ]

    for i in range(num_frames):
        # Create frame with solid color
        frame = np.full((height, width, 3), colors[i], dtype=np.uint8)

        # Add frame number text
        text = f"Frame {i+1}"
        cv2.putText(frame, text, (width//2 - 100, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        out.write(frame)

    out.release()
    print(f"Generated test video: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    generate_test_video()
