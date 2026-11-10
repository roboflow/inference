"""Use cameras already connected to Cloud Streaming.

ROBOFLOW_API_KEY=... python registered_source.py                    # list sources
ROBOFLOW_API_KEY=... python registered_source.py <source-id> <workflow-id>
ROBOFLOW_API_KEY=... python registered_source.py --attach <job-id>  # follow a deployment
"""

import sys

from inference_sdk.cloud_streaming import CloudStreamingClient


def main() -> None:
    client = CloudStreamingClient()
    if len(sys.argv) == 1:
        for source in client.list_sources():
            print(
                source["id"], source["kind"], source.get("status"), source.get("name")
            )
        return
    if sys.argv[1] == "--attach":
        stream = client.attach_stream(sys.argv[2])
    else:
        stream = client.start_stream(sys.argv[1], workflow=sys.argv[2])
    with stream:
        for result in stream.results():
            print(result.frame_id, result.outputs)


if __name__ == "__main__":
    main()
