"""Run a Workflow on this computer's webcam and show its annotated output.

pip install "inference-sdk[streaming]"
ROBOFLOW_API_KEY=... python webcam.py <workflow-id> <image-output>
"""

import sys
import threading

import cv2

from inference_sdk.cloud_streaming import CloudStreamingClient, WebcamSource


def main() -> None:
    workflow, image_output = sys.argv[1], sys.argv[2]
    client = CloudStreamingClient()
    with client.start_stream(WebcamSource(), workflow=workflow) as stream:
        threading.Thread(target=print_results, args=(stream,), daemon=True).start()
        for frame in stream.video(image_output):
            cv2.imshow("Cloud Streaming", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def print_results(stream) -> None:
    for result in stream.results():
        print(
            f"frame {result.frame_id}: {sorted(result.outputs)} ({result.latency_ms} ms)"
        )


if __name__ == "__main__":
    main()
