# WebRTC Webcam Streaming with Roboflow Inference

This example demonstrates how to stream webcam video through WebRTC to a Roboflow inference server for real-time workflow processing.

## Overview

The script captures video from your local webcam and streams it via WebRTC to an inference server, which processes the frames through workflow and returns the results. The processed video stream is displayed locally with the ability to dynamically switch between different workflow outputs using keyboard shortcuts.

## Prerequisites

- `inference` package

## Usage

```bash
python webcam.py \
  --workflow-id YOUR_WORKFLOW_ID \
  --workspace-id YOUR_WORKSPACE_ID \
  --inference-server-url http://localhost:9001 \
  --api-key YOUR_API_KEY
```

### Command Line Arguments

- `--workflow-id` (required): ID of the Roboflow workflow to run
- `--workspace-id` (required): Your Roboflow workspace ID
- `--inference-server-url` (required): URL of the inference server
- `--api-key` (required): Your Roboflow API key

## Keyboard Controls

While the stream is running, use these keyboard shortcuts:

### Video Output Selection
- `1-9`: Switch to workflow output 1-9 (video stream)
- `0`: Turn off video stream output

### Data Output Selection
- `a-j`: Switch to workflow output 1-10 (data channel)
- `z`: Turn off data output

### General
- `q`: Quit the application

## How It Works

1. **Webcam Capture**: The script captures frames from your local webcam using OpenCV
2. **WebRTC Connection**: Establishes a peer-to-peer connection with the inference server
3. **Stream Processing**: Sends video frames to the server for workflow processing
4. **Results Display**: Shows the processed video stream with workflow visualizations
5. **Data Channel**: Prints inference output to stdout
