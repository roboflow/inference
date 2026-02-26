import threading
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from inference.core.interfaces.camera.entities import (
    SourceProperties,
    VideoFrameProducer,
)

WIDTH = 1920
HEIGHT = 1080
FPS = 10
GRID_COLS = 16
GRID_ROWS = 9
BORDER_PX = 5
COLOR_SHUFFLE_INTERVAL = 5.0


def _generate_grid_colors(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(40, 220, size=(GRID_ROWS, GRID_COLS, 3), dtype=np.uint8)


def _draw_color_grid(frame: np.ndarray, colors: np.ndarray) -> None:
    cell_w = WIDTH // GRID_COLS
    cell_h = HEIGHT // GRID_ROWS
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = col * cell_w + BORDER_PX
            y1 = row * cell_h + BORDER_PX
            x2 = (col + 1) * cell_w - BORDER_PX
            y2 = (row + 1) * cell_h - BORDER_PX
            color = tuple(int(c) for c in colors[row, col])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)


def _draw_outlined_text(
    frame: np.ndarray,
    text: str,
    center_x: int,
    center_y: int,
    font_scale: float,
    thickness: int,
    outline: int,
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = center_x - tw // 2
    y = center_y + th // 2
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + outline, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_fixed_width_time(
    frame: np.ndarray,
    time_str: str,
    center_x: int,
    center_y: int,
    font_scale: float,
    thickness: int,
    outline: int,
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    (char_w, _), _ = cv2.getTextSize("0", font, font_scale, thickness)
    slot_w = int(char_w * 1.15)
    total_w = slot_w * len(time_str)
    start_x = center_x - total_w // 2

    for i, ch in enumerate(time_str):
        cx = start_x + i * slot_w + slot_w // 2
        if ch == ":":
            _draw_outlined_text(frame, ch, cx, center_y, font_scale, thickness, outline)
        else:
            _draw_outlined_text(frame, ch, cx, center_y, font_scale, thickness, outline)


def generate_frame(now: Optional[datetime] = None) -> np.ndarray:
    if now is None:
        now = datetime.now()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    color_seed = int(time.time() / COLOR_SHUFFLE_INTERVAL)
    colors = _generate_grid_colors(color_seed)
    _draw_color_grid(frame, colors)

    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    time_scale = 4.0
    date_scale = 1.5
    time_thickness = 6
    date_thickness = 2
    outline = 20

    cx = WIDTH // 2
    time_cy = HEIGHT // 2 - int(HEIGHT * 0.04)
    _draw_fixed_width_time(frame, time_str, cx, time_cy, time_scale, time_thickness, outline)

    font = cv2.FONT_HERSHEY_DUPLEX
    (_, time_h), _ = cv2.getTextSize("0", font, time_scale, time_thickness)
    date_cy = time_cy + time_h + int(HEIGHT * 0.06)
    _draw_outlined_text(frame, date_str, cx, date_cy, date_scale, date_thickness, outline)

    return frame


class _SharedGenerator:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ref_count = 0

    def acquire(self) -> None:
        with self._lock:
            self._ref_count += 1
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

    def release(self) -> None:
        with self._lock:
            self._ref_count = max(0, self._ref_count - 1)
            if self._ref_count == 0:
                self._running = False

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self._frame

    def _loop(self) -> None:
        interval = 1.0 / FPS
        while self._running:
            self._frame = generate_frame()
            time.sleep(interval)
        self._frame = None


_shared = _SharedGenerator()


class TestPatternStreamProducer(VideoFrameProducer):

    def __init__(self):
        self._opened = False

    def grab(self) -> bool:
        return self._opened and _shared.frame is not None

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        f = _shared.frame
        if f is None:
            return False, None
        return True, f.copy()

    def release(self) -> None:
        if self._opened:
            _shared.release()
            self._opened = False

    def isOpened(self) -> bool:
        if not self._opened:
            _shared.acquire()
            self._opened = True
            for _ in range(50):
                if _shared.frame is not None:
                    break
                time.sleep(0.05)
        return self._opened

    def discover_source_properties(self) -> SourceProperties:
        return SourceProperties(
            width=WIDTH,
            height=HEIGHT,
            total_frames=-1,
            is_file=False,
            fps=float(FPS),
            is_reconnectable=True,
        )

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        pass


REFERENCE_KEY = "TestPatternStreamProducer"


def resolve_test_pattern_reference(video_reference):
    if isinstance(video_reference, str) and video_reference.strip().startswith(REFERENCE_KEY):
        return lambda: TestPatternStreamProducer()
    return video_reference


if __name__ == "__main__":
    frame = generate_frame()
    out_path = "test_pattern_preview.png"
    cv2.imwrite(out_path, frame)
    print(f"Saved preview frame to {out_path} ({frame.shape[1]}x{frame.shape[0]})")
