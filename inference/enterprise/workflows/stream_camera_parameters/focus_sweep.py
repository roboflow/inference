import time
from dataclasses import dataclass
from typing import Optional, Tuple

from inference.enterprise.workflows.stream_camera_parameters.configure_client import (
    configure_usb_camera,
)
from inference.enterprise.workflows.stream_camera_parameters.entities import (
    ApplyCameraParametersResult,
)


@dataclass
class FocusSweepState:
    focus_value: int = 0
    last_tick_at: float = 0.0


def next_focus_value(current: int, step: int, max_focus: int) -> int:
    candidate = current + step
    if candidate > max_focus:
        return 0
    return candidate


def should_tick(last_tick_at: float, now: float, interval_seconds: float) -> bool:
    if last_tick_at <= 0:
        return True
    return (now - last_tick_at) >= interval_seconds


def run_focus_sweep_tick(
    state: FocusSweepState,
    *,
    interval_seconds: float,
    step: int,
    max_focus: int,
    video_reference: str,
    edge_base_url: str,
    lens_control: str = "lens_position",
    now: Optional[float] = None,
) -> Tuple[FocusSweepState, bool, ApplyCameraParametersResult | None]:
    current_time = now if now is not None else time.time()
    if not should_tick(state.last_tick_at, current_time, interval_seconds):
        return state, False, None

    next_focus = next_focus_value(state.focus_value, step, max_focus)
    result = configure_usb_camera(
        video_reference,
        {lens_control: next_focus},
        base_url=edge_base_url,
    )

    updated_state = FocusSweepState(
        focus_value=next_focus if result.success else state.focus_value,
        last_tick_at=current_time if result.success else state.last_tick_at,
    )
    return updated_state, True, result
