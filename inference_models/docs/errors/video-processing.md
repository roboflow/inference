# Local-video processing errors

These errors are exported by `inference_models`. The pipeline releases its decoder
on every error; the caller retains ownership of the loaded model.

## VideoInputError

The path does not name a local file, decoding cannot start, source FPS is invalid,
or fewer than two source frames are available. Check that the file exists and that
your OpenCV installation supports its codec. URLs are not accepted by the local
pipeline; download authorized inputs separately before passing a local path.

## VideoProcessingLimitError

Planning, prepared frame storage, candidates, timeline entries or encoded output
would exceed `VideoProcessingLimits`. The pipeline fails explicitly without
silently truncating data. A collected run returns no partial result. A stream
consumer may already have received earlier events.

Use `pipeline.stream()` for large output, select fewer classes, disable candidates,
split the input, or raise the relevant limit to fit your memory budget. Streaming
still enforces planning, frame-buffer and per-event limits.

## VideoProcessingCancelledError

The cancellation callback supplied to `VideoProcessingControl` returned true.
This is a caller-requested stop. Create a new control when restarting work.

## VideoProcessingTimeoutError

The control's deadline expired. The deadline starts when the control is created.
Use a longer budget or split the input. Cancellation and timeouts are cooperative:
an active model/device operation finishes before cleanup.
