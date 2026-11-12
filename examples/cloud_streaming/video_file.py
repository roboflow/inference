"""Play a video file into Cloud Streaming as a live source and print each frame's results.

pip install "inference-sdk[streaming]"
ROBOFLOW_API_KEY=... python video_file.py <path-to-video> <workflow-id>
"""

import sys

from inference_sdk.cloud_streaming import CloudStreamingClient, VideoFileSource


def main() -> None:
    path, workflow = sys.argv[1], sys.argv[2]
    client = CloudStreamingClient()
    with client.start_stream(
        VideoFileSource(path), workflow=workflow, max_fps=15
    ) as stream:
        for result in stream.results(
            on_gap=lambda gap: print(f"missed {gap.from_seq}-{gap.to_seq}")
        ):
            print(result.frame_id, result.outputs)


if __name__ == "__main__":
    main()
