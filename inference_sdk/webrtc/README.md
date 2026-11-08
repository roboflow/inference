# Session shutdown

`WebRTCSession.close()` closes the peer, cleans its source, and stops and joins
the session event loop. Calls from a WebRTC callback remain nonblocking; an
external call or context-manager exit waits for that shutdown result.

`WEBRTC_CLEANUP_TIMEOUT` controls the graceful wait for each asynchronous
startup-cancellation, peer-close and source-cleanup phase (default: 10 seconds).
If peer close exceeds that grace, the SDK aborts only this peer's owned ICE
transports and TURN carriers, then gives the original close task a bounded
chance to finish. Buffered outgoing data may be discarded at that point.
This does not change a running session's ICE configuration or disable IPv6.

`WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT` (default: 2 seconds) bounds cancellation,
fallback completion and the final loop-thread join. Both settings must be
finite and positive. A failed cleanup raises `TimeoutError` or its underlying
error; repeated external close callers observe that failure too. Background
callback cleanup failures are logged and can be observed by an external close.
If several cleanup phases fail, the first error is preserved and later errors
are logged. Task and asynchronous-generator finalizer failures are also visible.

These are asynchronous cleanup budgets, not a hard deadline for arbitrary
synchronous user code or the existing startup HTTP request. A nonresponsive
event loop or cancellation-resistant resource is reported as incomplete
cleanup. A forced local TURN carrier close does not establish that the remote
allocation was gracefully deleted.

The fallback's aioice-specific transport lookup is isolated in `_cleanup.py`.
It follows only the peer's SCTP/transceiver ICE connections; it never scans
other sessions, the process or the operating system for sockets.
