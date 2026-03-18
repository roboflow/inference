import logging
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Set V4L2 camera controls in a single pass"
LONG_DESCRIPTION = """
Configure V4L2 USB camera controls (exposure, gain, focus, LEDs, etc.) in one block.

Each input corresponds to a specific V4L2 driver control. Only non-None inputs are
applied — omit a field to leave it unchanged. Ranges are validated in Python before
calling v4l2-ctl.

Control names and ranges in this block target specific industrial USB cameras. If your
camera uses different control names or ranges, the v4l2-ctl error will be reported
in the block output.

## Requirements

- `v4l-utils` package must be installed in the container
- Container must have access to `/dev/video*` devices (privileged mode or device mapping)
"""

DEVICE_PATH_PATTERN = re.compile(r"^/dev/video\d+$")

V4L2ControlField = Optional[Union[Selector(kind=[INTEGER_KIND]), int]]

CONTROL_MAP: Dict[str, Tuple[str, int, int]] = {
    "auto_exposure": ("auto_exposure", 0, 1),
    "exposure_time": ("exposure_t", 1, 10000),
    "auto_exposure_max": ("auto_exposure_max", 1, 10000),
    "auto_gain": ("auto_gain", 0, 1),
    "gain": ("gain", 0, 24),
    "auto_gain_max": ("auto_gain_max", 0, 24),
    "brightness": ("brightness", 0, 255),
    "focus_direction": ("focus_direction", 0, 1),
    "focus_distance": ("focus_dist", 0, 2294),
    "led_enable": ("led_op", 0, 1),
    "led_select": ("led_position", 0, 7),
    "led_brightness": ("led_gain", 0, 100),
    "gamma": ("gamma_value", 0, 400),
    "sharpness": ("sharpness_value", 0, 100),
}

logger = logging.getLogger(__name__)


def _validate_device_path(device_path: str) -> None:
    if not DEVICE_PATH_PATTERN.match(device_path):
        raise ValueError(
            f"Invalid device path: {device_path}. Expected format: /dev/videoN"
        )


def _run_v4l2_ctl(args: List[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            ["v4l2-ctl"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError("v4l2-ctl not found. Install v4l-utils package.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("v4l2-ctl command timed out")
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "No such file or directory" in stderr or "No such device" in stderr:
            raise RuntimeError(f"Device not found: {stderr}")
        if "Permission denied" in stderr:
            raise RuntimeError(
                "Permission denied. Container may need privileged mode."
            )
        raise RuntimeError(f"v4l2-ctl error: {stderr}")
    return result.stdout


def _set_control(device: str, control_name: str, value: int) -> None:
    _run_v4l2_ctl(["-d", device, "--set-ctrl", f"{control_name}={value}"])


class V4L2CameraControlManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "V4L2 Camera Control",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-camera",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/v4l2_camera_control@v1"]
    device_path: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Device path, e.g. /dev/video0",
    )
    auto_exposure: V4L2ControlField = Field(
        default=None,
        description="Auto exposure mode: 0=manual, 1=auto (range 0-1)",
    )
    exposure_time: V4L2ControlField = Field(
        default=None,
        description="Manual exposure time (range 1-10000)",
    )
    auto_exposure_max: V4L2ControlField = Field(
        default=None,
        description="Upper limit for auto exposure (range 1-10000)",
    )
    auto_gain: V4L2ControlField = Field(
        default=None,
        description="Auto gain mode: 0=manual, 1=auto (range 0-1)",
    )
    gain: V4L2ControlField = Field(
        default=None,
        description="Sensor gain (range 0-24)",
    )
    auto_gain_max: V4L2ControlField = Field(
        default=None,
        description="Upper limit for auto gain (range 0-24)",
    )
    brightness: V4L2ControlField = Field(
        default=None,
        description="Image brightness (range 0-255)",
    )
    focus_direction: V4L2ControlField = Field(
        default=None,
        description="Focus direction: 0=near, 1=far (range 0-1)",
    )
    focus_distance: V4L2ControlField = Field(
        default=None,
        description="Focus motor steps (range 0-2294)",
    )
    led_enable: V4L2ControlField = Field(
        default=None,
        description="LED on/off: 0=off, 1=on (range 0-1)",
    )
    led_select: V4L2ControlField = Field(
        default=None,
        description="Which LED to control (range 0-7)",
    )
    led_brightness: V4L2ControlField = Field(
        default=None,
        description="LED brightness percentage (range 0-100)",
    )
    gamma: V4L2ControlField = Field(
        default=None,
        description="Gamma correction value (range 0-400)",
    )
    sharpness: V4L2ControlField = Field(
        default=None,
        description="Sharpness enhancement (range 0-100)",
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Minimum seconds between v4l2-ctl invocations (0 to disable)",
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Flag to disable this sink at runtime",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="success", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="error", kind=[STRING_KIND]),
            OutputDefinition(name="controls_set", kind=[INTEGER_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class V4L2CameraControlBlockV1(WorkflowBlock):

    def __init__(self):
        self._last_invoked: Optional[datetime] = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return V4L2CameraControlManifest

    def run(
        self,
        device_path: str,
        cooldown_seconds: int = 0,
        disable_sink: bool = False,
        auto_exposure: Optional[int] = None,
        exposure_time: Optional[int] = None,
        auto_exposure_max: Optional[int] = None,
        auto_gain: Optional[int] = None,
        gain: Optional[int] = None,
        auto_gain_max: Optional[int] = None,
        brightness: Optional[int] = None,
        focus_direction: Optional[int] = None,
        focus_distance: Optional[int] = None,
        led_enable: Optional[int] = None,
        led_select: Optional[int] = None,
        led_brightness: Optional[int] = None,
        gamma: Optional[int] = None,
        sharpness: Optional[int] = None,
    ) -> BlockResult:
        if disable_sink:
            return {
                "success": True,
                "throttling_status": False,
                "error": "",
                "controls_set": 0,
            }
        if cooldown_seconds > 0 and self._last_invoked is not None:
            elapsed = (datetime.now() - self._last_invoked).total_seconds()
            if elapsed < cooldown_seconds:
                logger.info("V4L2 camera control cooldown active, skipping")
                return {
                    "success": True,
                    "throttling_status": True,
                    "error": "",
                    "controls_set": 0,
                }
        try:
            _validate_device_path(device_path)
        except ValueError as e:
            return {
                "success": False,
                "throttling_status": False,
                "error": str(e),
                "controls_set": 0,
            }

        params = {
            "auto_exposure": auto_exposure,
            "exposure_time": exposure_time,
            "auto_exposure_max": auto_exposure_max,
            "auto_gain": auto_gain,
            "gain": gain,
            "auto_gain_max": auto_gain_max,
            "brightness": brightness,
            "focus_direction": focus_direction,
            "focus_distance": focus_distance,
            "led_enable": led_enable,
            "led_select": led_select,
            "led_brightness": led_brightness,
            "gamma": gamma,
            "sharpness": sharpness,
        }

        errors: List[str] = []
        controls_set = 0
        for field_name, (v4l2_name, min_val, max_val) in CONTROL_MAP.items():
            value = params[field_name]
            if value is None:
                continue
            if value < min_val or value > max_val:
                errors.append(
                    f"{field_name}: value {value} out of range [{min_val}, {max_val}]"
                )
                continue
            try:
                _set_control(device_path, v4l2_name, value)
                controls_set += 1
            except Exception as e:
                errors.append(f"{field_name}: {e}")
        self._last_invoked = datetime.now()
        if errors:
            logger.warning("V4L2 control errors: %s", "; ".join(errors))
        else:
            logger.debug("V4L2: set %d controls on %s", controls_set, device_path)
        return {
            "success": len(errors) == 0,
            "throttling_status": False,
            "error": "; ".join(errors),
            "controls_set": controls_set,
        }
