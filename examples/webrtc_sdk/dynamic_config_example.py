"""
WebRTC SDK example demonstrating dynamic channel configuration.

This example shows how to change stream and data outputs in real-time
during an active WebRTC session without reconnecting. Uses a workflow
specification directly (no need for workspace/workflow-id).

Usage:
  python examples/webrtc_sdk/dynamic_config_example.py \
      [--api-url http://localhost:9001] \
      [--api-key <ROBOFLOW_API_KEY>] \
      [--width 1920] \
      [--height 1080]

Controls:
  q - Quit
  + - Enable all data outputs
  - - Disable all data outputs
  a-z - Toggle individual data outputs
  0 - Disable video output
  1-9 - Switch video output

The example uses a workflow specification defined in the code, so no need
for workspace/workflow-id parameters. Press keys in the preview window to
dynamically control which outputs are sent.
"""
import argparse
import json

import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoMetadata, WebcamSource, StreamConfig

# Example workflow specification
# This is a simple workflow that runs object detection and provides outputs
WORKFLOW_SPEC_JSON = """{
  "version": "1.0",
  "inputs": [
    {
      "type": "InferenceImage",
      "name": "image"
    }
  ],
  "steps": [
    {
      "type": "roboflow_core/relative_statoic_crop@v1",
      "name": "relative_static_crop",
      "images": "$inputs.image",
      "x_center": 0.5,
      "y_center": 0.5,
      "width": 0,
      "height": 0.5
    },
    {
      "type": "roboflow_core/property_definition@v1",
      "name": "property_definition",
      "data": "$inputs.image",
      "operations": [
        {
          "type": "ExtractImageProperty",
          "property_name": "aspect_ratio"
        }
      ]
    },
    {
      "type": "roboflow_core/image_blur@v1",
      "name": "image_blur",
      "image": "$inputs.image"
    }
  ],
  "outputs": [
    {
      "type": "JsonField",
      "name": "image_blur",
      "coordinates_system": "own",
      "selector": "$steps.image_blur.image"
    },
    {
      "type": "JsonField",
      "name": "image",
      "coordinates_system": "own",
      "selector": "$steps.relative_static_crop.crops"
    },
    {
      "type": "JsonField",
      "name": "original_ratio",
      "coordinates_system": "own",
      "selector": "$steps.property_definition.output"
    }
  ]
}"""

# Parse the JSON specification into a Python dict
WORKFLOW_SPEC = json.loads(WORKFLOW_SPEC_JSON)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("WebRTC SDK Dynamic Configuration Example")
    p.add_argument("--api-url", default="http://localhost:9001")
    p.add_argument("--image-input-name", default="image")
    p.add_argument("--api-key", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)

    # Extract available outputs from workflow specification
    workflow_outputs = WORKFLOW_SPEC.get("outputs", [])
    available_output_names = [o.get("name") for o in workflow_outputs]

    if not workflow_outputs:
        print("⚠️  Workflow has no outputs defined")
        return

    print(f"Available workflow outputs: {available_output_names}")

    # Prepare source
    resolution = None
    if args.width and args.height:
        resolution = (args.width, args.height)
    source = WebcamSource(resolution=resolution)

    # Start with some outputs configured
    config = StreamConfig(
        stream_output=[available_output_names[0]] if available_output_names else [],  # Use first output
        data_output=[]  # Start with no data outputs
    )

    # Start streaming session with workflow specification
    session = client.webrtc.stream(
        source=source,
        workflow=WORKFLOW_SPEC,  # Pass workflow spec directly
        image_input=args.image_input_name,
        config=config,
    )

    with session:
        # Track current configuration state for display
        current_data_mode = "none"
        active_data_fields = []  # For custom mode

        def draw_output_list(frame):
            """Draw list of available outputs with active indicators"""
            x_start = 10
            y_start = 80
            line_height = 22

            # Title
            if current_data_mode == "all":
                title = "Data Outputs (ALL)"
                title_color = (100, 255, 100)
            elif current_data_mode == "none":
                title = "Data Outputs (NONE)"
                title_color = (100, 100, 100)
            else:
                title = f"Data Outputs ({len(active_data_fields)} active)"
                title_color = (100, 200, 255)

            cv2.putText(frame, title, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_color, 1, cv2.LINE_AA)
            y_start += line_height + 5

            # Draw each output
            for i, output in enumerate(workflow_outputs):
                key_letter = chr(ord("a") + i) if i < 26 else "?"
                output_name = output.get("name", "unnamed")

                # Determine if active
                if current_data_mode == "all":
                    is_active = True
                elif current_data_mode == "none":
                    is_active = False
                else:
                    is_active = output_name in active_data_fields

                # Format line with ASCII checkbox
                indicator = "[X]" if is_active else "[ ]"
                color = (100, 255, 100) if is_active else (100, 100, 100)
                text = f"  [{key_letter}] {indicator} {output_name}"

                cv2.putText(
                    frame,
                    text,
                    (x_start, y_start + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )

            # Controls
            y_controls = y_start + len(workflow_outputs) * line_height + 10
            cv2.putText(
                frame,
                "  [+] All  [-] None  [1-9] Video Output",
                (x_start, y_controls),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        @session.on_frame
        def show_frame(frame, metadata):
            nonlocal current_data_mode, active_data_fields

            # Draw output list overlay
            draw_output_list(frame)

            # Add controls hint at bottom
            controls = "q=quit | +=all | -=none | a-z=toggle data | 0-9=video"
            cv2.putText(
                frame,
                controls,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )

            cv2.imshow("WebRTC SDK - Dynamic Configuration", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting...")
                session.stop()

            elif key == ord("+") or key == ord("="):
                print("Setting data output to ALL")
                session.set_data_outputs(None)
                current_data_mode = "all"

            elif key == ord("-"):
                print("Setting data output to NONE")
                session.set_data_outputs([])
                current_data_mode = "none"

            elif key == ord("0"):
                print("Disabling video output")
                session.set_stream_output("")

            # Handle 1-9 keys for video output selection
            elif ord("1") <= key <= ord("9"):
                output_index = key - ord("1")
                if output_index < len(available_output_names):
                    output_name = available_output_names[output_index]
                    print(f"Switching video to '{output_name}'")
                    session.set_stream_output(output_name)

            # Handle a-z keys for data output toggling
            elif chr(key).isalpha() and chr(key).lower() in "abcdefghijklmnopqrstuvwxyz":
                key_index = ord(chr(key).lower()) - ord("a")
                if key_index < len(workflow_outputs):
                    output_name = workflow_outputs[key_index].get("name", "")

                    # Toggle logic
                    if current_data_mode == "all":
                        # Was "all", switch to custom with all except this one
                        current_data_mode = "custom"
                        active_data_fields = list(available_output_names)
                        active_data_fields.remove(output_name)
                        print(f"Toggled OFF '{output_name}' (was ALL)")
                    elif current_data_mode == "none":
                        # Was "none", enable only this field
                        current_data_mode = "custom"
                        active_data_fields = [output_name]
                        print(f"Toggled ON '{output_name}' (was NONE)")
                    else:
                        # Custom mode - toggle
                        if output_name in active_data_fields:
                            active_data_fields.remove(output_name)
                            print(f"Toggled OFF '{output_name}'")
                        else:
                            active_data_fields.append(output_name)
                            print(f"Toggled ON '{output_name}'")

                    # Send updated list
                    print(f"Active fields: {active_data_fields}")
                    session.set_data_outputs(active_data_fields if active_data_fields else [])

        # Global data handler to monitor what we're receiving
        @session.on_data()
        def handle_data(data: dict, metadata: VideoMetadata):
            if data:
                print(f"Frame {metadata.frame_id}: Received fields: {list(data.keys())}")
            else:
                print(f"Frame {metadata.frame_id}: No data (metadata only)")

        # Run the session (blocks until stop() is called or stream ends)
        print("\n=== WebRTC Dynamic Configuration Example ===")
        print(f"Available outputs: {available_output_names}")
        print("\nControls:")
        print("  q     - Quit")
        print("  +     - Enable all data outputs")
        print("  -     - Disable all data outputs (metadata only)")
        for i, output in enumerate(workflow_outputs):
            key_letter = chr(ord("a") + i) if i < 26 else "?"
            print(f"  {key_letter}     - Toggle '{output.get('name')}' data output")
        print("  0     - Disable video output")
        for i, name in enumerate(available_output_names[:9]):
            print(f"  {i+1}     - Switch video to '{name}'")
        print("\nPress keys in the video window to control outputs dynamically.\n")

        session.run()


if __name__ == "__main__":
    main()
