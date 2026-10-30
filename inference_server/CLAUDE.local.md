# inference-server

FastAPI HTTP layer on top of inference-model-manager. Owns endpoints, auth, response formatting, server lifecycle.

`from __future__ import annotations` breaks any FastAPI handler wrapped by with_legacy_errors, since FastAPI resolves forward refs against the wrapper's module globals, turning every body and path param into a query param
 - Never use `from __future__ import annotations` in router modules: FastAPI resolves forward refs through decorator wrappers' globals and silently turns body/path params into query params.
