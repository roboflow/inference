"""Selectively exported active blocks for the opt-in V2 catalogue."""



from roboflow_workflows.v2.blocks.industrial.onvif_sink.v1 import OnvifTrackingBlock




BLOCKS = (
    OnvifTrackingBlock,
)

__all__ = [
    "BLOCKS",
    "OnvifTrackingBlock",
]
