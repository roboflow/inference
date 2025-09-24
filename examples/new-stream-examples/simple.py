import argparse
import signal
import sys
from functools import partial
from typing import Optional, Union

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

PIPELINE: Optional[InferencePipeline] = None


def signal_handler(sig, frame):
    print("Terminating")
    if PIPELINE is not None:
        PIPELINE.terminate()
        PIPELINE.join()
    sys.exit(0)


def main(model_id: str, source: Union[str, int]) -> None:
    global PIPELINE
    PIPELINE = InferencePipeline.init(
        model_id=model_id,
        video_reference=source,
        on_prediction=partial(render_boxes, display_statistics=True),
    )
    PIPELINE.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simple InferencePipeline demo")
    parser.add_argument(
        "--model_id",
        help=f"ID of the model",
        required=False,
        type=str,
        default="rock-paper-scissors-sxsw/11",
    )
    parser.add_argument(
        "--source",
        help=f"Reference to video source - can be file, stream or id of device",
        required=False,
        default=0,
    )
    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C to terminate")
    args = parser.parse_args()
    main(model_id=args.model_id, source=args.source)
