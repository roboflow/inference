# Profiling API Requests with pyinstrument

The inference server supports detailed performance profiling of API requests using pyinstrument, which offers a deeper insight into the execution time of HTTP requests and models through statistical sampling.

## Enabling profiling

Install the profiling dependencies using pip:

```bash
pip install -r requirements/requirements.profiling.txt
```

Then set the environment variable `PYINSTRUMENT_ENABLED` to `true`:

```bash
export PYINSTRUMENT_ENABLED=true
```

## Profiling a request

To enable profiling for a specific request, simply append the query parameter `?profile=true` to the URL of any request. For example, to profile a Grounding DINO inference request:

```
curl -X POST -H "Content-Type: application/json" -d "@/home/user/request-data.json" "https://inference-url/grounding_dino/infer?profile=true"
```

## Viewing performance profiles

The inference server writes two files when a request is profiled:

- `profile.html:` A detailed HTML report of the profile.
    - Open in a Web browser.
- `profile.speedscope.json`: A JSON file containing the raw profile data in the Speedscope format.
    - Open with https://www.speedscope.app/.
