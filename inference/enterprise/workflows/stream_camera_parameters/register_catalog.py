from typing import Any, Callable, Dict, List, Optional

REGISTER_LABELS: Dict[str, str] = {
    "focus": "Focus (lens position)",
    "line_rate": "Line rate",
    "exposure_time": "Exposure time",
    "brightness": "Brightness",
    "width": "Width",
    "height": "Height",
    "exposure_lower": "Exposure lower limit",
    "exposure_upper": "Exposure upper limit",
    "gain_lower": "Gain lower limit",
    "gain_upper": "Gain upper limit",
    "timeout": "Grab / heartbeat timeout",
    "lines_per_frame": "Lines per frame",
    "fps": "FPS",
    "manual": "Manual register",
}

CAMERA_FAMILIES = ("usb", "ai1", "basler", "basler_line_scan", "lucid", "lucid_line_scan")

ParameterBuilder = Callable[[Any], Dict[str, Any]]

REGISTER_BUILDERS: Dict[str, Dict[str, ParameterBuilder]] = {
    "focus": {
        "ai1": lambda value: {"v4l2_camera_properties": {"lens_position": value}},
        "usb": lambda value: {"v4l2_camera_properties": {"lens_position": value}},
    },
    "line_rate": {
        "basler_line_scan": lambda value: {"AcquisitionLineRate": value},
        "lucid_line_scan": lambda value: {"AcquisitionLineRate": value},
    },
    "exposure_time": {
        "ai1": lambda value: {
            "v4l2_camera_properties": {"exposure_mode": 0, "exposure_time": value}
        },
        "basler_line_scan": lambda value: {"ExposureTime": value},
        "lucid_line_scan": lambda value: {"ExposureTime": value},
    },
    "lines_per_frame": {
        "basler_line_scan": lambda value: {"LinesPerFrame": value},
        "lucid_line_scan": lambda value: {"LinesPerFrame": value},
    },
    "brightness": {
        "basler": lambda value: {"TARGET_BRIGHTNESS": value},
        "basler_line_scan": lambda value: {"TARGET_BRIGHTNESS": value},
        "lucid": lambda value: {"TargetBrightness": value},
        "lucid_line_scan": lambda value: {"TargetBrightness": value},
        "usb": lambda value: {"v4l2_camera_properties": {"brightness": value}},
        "ai1": lambda value: {"v4l2_camera_properties": {"brightness": value}},
    },
    "width": {
        "basler": lambda value: {"pylon_nodemap": {"Width": value}},
        "basler_line_scan": lambda value: {"pylon_nodemap": {"Width": value}},
        "lucid": lambda value: {"Width": value},
        "lucid_line_scan": lambda value: {"Width": value},
        "usb": lambda value: {"video_source_properties": {"frame_width": value}},
        "ai1": lambda value: {"video_source_properties": {"frame_width": value}},
    },
    "height": {
        "basler": lambda value: {"pylon_nodemap": {"Height": value}},
        "lucid": lambda value: {"Height": value},
        "lucid_line_scan": lambda value: {"Height": value},
        "usb": lambda value: {"video_source_properties": {"frame_height": value}},
        "ai1": lambda value: {"video_source_properties": {"frame_height": value}},
    },
    "exposure_lower": {
        "basler": lambda value: {"EXPOSURE_LOWER_LIMIT": value},
        "basler_line_scan": lambda value: {"EXPOSURE_LOWER_LIMIT": value},
        "lucid": lambda value: {"ExposureAutoLowerLimit": value},
        "lucid_line_scan": lambda value: {"ExposureAutoLowerLimit": value},
    },
    "exposure_upper": {
        "basler": lambda value: {"EXPOSURE_UPPER_LIMIT": value},
        "basler_line_scan": lambda value: {"EXPOSURE_UPPER_LIMIT": value},
        "lucid": lambda value: {"ExposureAutoUpperLimit": value},
        "lucid_line_scan": lambda value: {"ExposureAutoUpperLimit": value},
    },
    "gain_lower": {
        "basler": lambda value: {"GAIN_LOWER_LIMIT": value},
        "basler_line_scan": lambda value: {"GAIN_LOWER_LIMIT": value},
    },
    "gain_upper": {
        "basler": lambda value: {"GAIN_UPPER_LIMIT": value},
        "basler_line_scan": lambda value: {"GAIN_UPPER_LIMIT": value},
    },
    "timeout": {
        "basler": lambda value: {"GEV_HEARTBEAT_TIMEOUT": value},
        "basler_line_scan": lambda value: {"GEV_HEARTBEAT_TIMEOUT": value},
        "lucid": lambda value: {"grab_timeout_ms": value},
        "lucid_line_scan": lambda value: {"grab_timeout_ms": value},
    },
    "fps": {
        "usb": lambda value: {"video_source_properties": {"fps": value}},
        "ai1": lambda value: {"video_source_properties": {"fps": value}},
    },
}


def registers_for_camera_family(camera_family: str) -> List[str]:
    family = (camera_family or "").strip().lower()
    if family not in CAMERA_FAMILIES:
        return ["manual"]
    supported = [
        register
        for register, families in REGISTER_BUILDERS.items()
        if family in families
    ]
    supported.append("manual")
    return supported


def build_parameter_delta(
    register: str,
    value: Any,
    *,
    camera_family: str,
    manual_register_key: Optional[str] = None,
) -> Dict[str, Any]:
    family = (camera_family or "").strip().lower()
    register_key = (register or "").strip().lower()

    if register_key == "manual":
        key = (manual_register_key or "").strip()
        if not key:
            raise ValueError("manual_register_key is required when register is manual")
        if not family:
            raise ValueError("camera_family is required for manual register writes")
        return {key: value}

    if not family:
        raise ValueError("camera_family is required")

    builder = REGISTER_BUILDERS.get(register_key, {}).get(family)
    if builder is None:
        raise ValueError(
            f"Register '{register_key}' is not supported for camera family '{family}'"
        )

    return builder(value)
