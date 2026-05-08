# ROS 2 Interop Demo

End-to-end example: drive a Roboflow Workflow from a ROS image topic and
publish detections back onto ROS, all over the rosbridge JSON-WebSocket
protocol — no ROS install required on the inference side.

## What's in here

| File | Purpose |
| --- | --- |
| `ros_workflow_demo.ipynb` | Step-by-step notebook (this is the entry point). |
| `inference.rviz` | RViz 2 config that shows the raw camera image, the annotated overlay, and the `Detection2DArray` topic. |

## Prerequisites

- Linux host with Docker. The notebook runs `rosbridge_server` and `rviz2`
  in containers so you don't need ROS on the host.
- An inference server with the `ros` extra installed and reachable from
  the rosbridge container. The dev image already has it
  (`docker run --network host roboflow/roboflow-inference-server-gpu:dev`),
  or build your own with `pip install 'inference[ros]'`.
- A Roboflow API key for `rfdetr-medium` weight access.
- Free TCP ports `9090` (rosbridge) and `9001` (inference HTTP API).

## Architecture

```
  ┌────────────────┐   rosbridge://camera/image_raw/compressed
  │ video publisher├────────────────────────┐
  │   (notebook)   │                        ▼
  └────────────────┘            ┌──────────────────────┐
                                │   rosbridge_server   │ ◄── RViz 2
                                │   (ws://:9090)       │     subscribes
                                └─▲──────────────────▲─┘     to all topics
                                  │ subscribe        │ publish
                                  │ image            │ /inference/detections
                                  │                  │ /inference/image_annotated
                                ┌─┴──────────────────┴─┐
                                │  Inference server    │
                                │  (workflow runtime,  │
                                │   roslibpy client)   │
                                └──────────────────────┘
```

Both the inference server and `rosbridge_server` need to share a network so
they can reach each other on `localhost:9090`. The notebook uses
`--network host` for everything; if you're on a setup where that's not an
option, swap in a shared user-defined Docker network.

## Quick run

```bash
cd examples/ros
jupyter lab ros_workflow_demo.ipynb
```

Then walk through the cells top-to-bottom.
