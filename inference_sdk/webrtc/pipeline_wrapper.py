"""
WebRTC Pipeline Wrapper for callback-based workflow processing.

Provides an InferencePipeline-like interface for WebRTC streaming with
traditional callback patterns.
"""

import base64
import logging
from threading import Thread
from typing import Callable, Optional

import cv2
import numpy as np

from inference_sdk.webrtc.session import VideoMetadata, WebRTCSession

logger = logging.getLogger(__name__)


class WebRTCPipelineWrapper:
    """
    Wraps WebRTCSession to provide callback-based interface similar to InferencePipeline.

    This wrapper bridges the new decorator-based WebRTC API with the traditional
    callback pattern, where a single `on_prediction` function is called for each
    processed frame.

    Example:
        ```python
        def my_callback(data: dict, metadata: VideoMetadata):
            # data contains decoded image + workflow outputs
            cv2.imshow("Frame", data["image"])
            print(f"Predictions: {data}")

        wrapper = WebRTCPipelineWrapper(
            session=session,
            on_prediction=my_callback,
            image_output_key="image"
        )
        wrapper.start()  # Non-blocking, runs in background
        # ... do other work ...
        wrapper.terminate()  # Stop and cleanup
        ```

    Args:
        session: WebRTCSession instance to wrap
        on_prediction: Callback function(data: dict, metadata: VideoMetadata)
        image_output_key: Key name for image output in workflow data
    """

    def __init__(
        self,
        session: WebRTCSession,
        on_prediction: Callable[[dict, VideoMetadata], None],
        image_output_key: str = "image",
    ):
        """Initialize the wrapper and register internal handlers.

        Args:
            session: WebRTCSession to wrap
            on_prediction: User callback for predictions
            image_output_key: Name of image field in workflow output
        """
        self._session = session
        self._on_prediction = on_prediction
        self._image_output_key = image_output_key
        self._thread: Optional[Thread] = None
        self._running = False

        # Register internal data handler to bridge to user callback
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register internal handlers on the WebRTC session."""

        @self._session.on_data()
        def _internal_data_handler(data: dict, metadata: VideoMetadata) -> None:
            """
            Internal handler that processes data and calls user callback.

            Decodes base64 image from data channel and invokes user's
            on_prediction callback with the processed data.
            """
            try:
                # Decode base64 image to numpy array
                processed_data = self._decode_image_from_data(data)

                # Call user's callback with processed data
                self._on_prediction(processed_data, metadata)
            except Exception as e:
                # Log error but don't crash the pipeline
                logger.error(
                    f"Error in on_prediction callback: {e}",
                    exc_info=True,
                )

    def _decode_image_from_data(self, data: dict) -> dict:
        """
        Decode base64 image from workflow output data.

        Args:
            data: Dictionary containing workflow outputs, including base64 image

        Returns:
            Dictionary with decoded numpy array image + other outputs
        """
        result = data.copy()

        # Find and decode image if present
        if self._image_output_key in data:
            base64_str = data[self._image_output_key]

            try:
                # Decode base64 -> bytes -> numpy array
                image_bytes = base64.b64decode(base64_str)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    logger.warning(
                        f"Failed to decode image from {self._image_output_key}"
                    )
                else:
                    result[self._image_output_key] = image
            except Exception as e:
                logger.error(
                    f"Error decoding image from {self._image_output_key}: {e}",
                    exc_info=True,
                )

        return result

    def start(self) -> None:
        """
        Start processing frames in background thread.

        This method returns immediately and runs session.run() in a
        background thread, matching the behavior of InferencePipeline.start().

        Raises:
            RuntimeError: If pipeline is already running
        """
        if self._running:
            raise RuntimeError("Pipeline is already running")

        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True, name="WebRTCPipeline")
        self._thread.start()
        logger.info("WebRTC pipeline started in background thread")

    def _run_loop(self) -> None:
        """
        Main loop for background thread.

        Runs session.run() which blocks until session is closed or stream ends.
        """
        try:
            # This blocks until session.close() is called or stream ends
            self._session.run()
        except Exception as e:
            logger.error(f"Error in WebRTC session: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info("WebRTC pipeline stopped")

    def terminate(self) -> None:
        """
        Stop the pipeline and cleanup resources.

        This method closes the WebRTC session and waits for the background
        thread to finish. Safe to call multiple times.
        """
        if not self._running:
            logger.debug("Pipeline already stopped or not started")
            return

        logger.info("Terminating WebRTC pipeline...")

        # Close session (this will cause run() to exit)
        self._session.close()

        # Wait for thread to finish with timeout
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "Background thread did not stop within timeout, "
                    "but resources have been cleaned up"
                )

        self._running = False
        logger.info("WebRTC pipeline terminated")

    def is_running(self) -> bool:
        """Check if pipeline is currently running.

        Returns:
            True if pipeline is running, False otherwise
        """
        return self._running

    def __enter__(self):
        """Context manager entry - starts the pipeline."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - terminates the pipeline."""
        self.terminate()
        return False
