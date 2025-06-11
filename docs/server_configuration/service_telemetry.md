# Inference server telemetry

Service telemetry provides essential real-time data on system health, performance, and usage. It enables:

* **Monitoring and Diagnostics:** Early detection of issues for quick resolution.

* **Performance Optimization:** Identifying bottlenecks to improve efficiency.

* **Usage Insights:** Understanding user behavior to guide improvements.

* **Security:** Detecting suspicious activities and ensuring compliance.

* **Scalability:** Predicting and managing resource demands.

In `inference` server, we enabled:

* [`prometheus`](https://prometheus.io/) metrics

* docker container metrics provided by Docker daemon

## 🔥 [`prometheus`](https://prometheus.io/) in `inference` server

To enable metrics, set environmental variable `ENABLE_PROMETHEUS=True` in your docker container:

```bash
docker run -p 9001:9001 -e ENABLE_PROMETHEUS=True roboflow/roboflow-inference-server-cpu
```

Then use `GET /metrics` endpoint to fetch the metrics in Python:

```python
import requests

result = requests.get("http://127.0.0.1:9001/metrics")
result.raise_for_status()

print(result.text)
```

or using curl:
```bash
curl http://127.0.0.1:9001/metrics
```

## Docker container metrics

!!! warning "Potential security issue"

    This feature rely on docker daemon socket exposure inside container. This way we can expose docker
    container resource utilisation metrics without the need for "supervisor" service, but may be seen
    as security violation. We disable that ooption by default. Please acknowledge 
    [potential security risks](https://www.lvh.io/posts/dont-expose-the-docker-socket-not-even-to-a-container/)
    before enabling this option

To expose container metrics you need to run docker container with `inference` server in a specific way:

```{ .bash linenums="1" hl_lines="2 3" }
docker run -p 9001:9001 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e DOCKER_SOCKET_PATH=/var/run/docker.sock
  roboflow/roboflow-inference-server-cpu
```

Explanation:

* In line `2`, you **mount** a docker daemon socket from your host (typically `/var/run/docker.sock`, 
but you need to verify your setup) into some location inside container (for convenience also `/var/run/docker.sock`)

* In line `3`, you expose environmental variable `DOCKER_SOCKET_PATH` with the location of the docker daemon socket 
**inside the container** - in this case `/var/run/docker.sock`, as per specification in line `2`

Then you would be able to reach `GET /device/stats` endpoint using cURL:
```bash
curl http://127.0.0.1:9001/device/stats
```

or using Python:
```python
import requests

result = requests.get("http://127.0.0.1:9001/device/stats")
result.raise_for_status()

print(result.json())
```
