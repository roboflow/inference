# API Performance Benchmark Tool

This tool compares the performance of the old (base64 JSON) and new (multipart) inference APIs.

## Installation

Install required dependencies:

```bash
pip install requests pillow psutil
```

## Usage

### Basic Usage

```bash
python -m tests.benchmark.api_performance \
    --host http://localhost:9001 \
    --api-key YOUR_API_KEY \
    --model-id your-project/version \
    --image-path /path/to/test/image.jpg
```

### Options

- `--host`: Inference server URL (default: `http://localhost:9001`)
- `--api-key`: Your Roboflow API key (required)
- `--model-id`: Model ID to test (required, e.g., `my-project/1`)
- `--image-path`: Path to test image (required)
- `--runs`: Number of runs per API (default: 10)
- `--confidence`: Confidence threshold (default: 0.5)

### Example

```bash
python -m tests.benchmark.api_performance \
    --host http://localhost:9001 \
    --api-key abc123def456 \
    --model-id construction-safety/3 \
    --image-path test_images/construction_site.jpg \
    --runs 20 \
    --confidence 0.6
```

## What It Measures

The benchmark tool measures:

1. **Base64 Encoding Time**: Time spent encoding images to base64 (old API only)
2. **Payload Size**: Request body size in KB/MB
3. **Request Time**: End-to-end time for each request including:
   - Network overhead
   - Request parsing
   - Inference execution
   - Response serialization
4. **Memory Usage**: Memory delta during request processing
5. **Statistical Analysis**: Mean, median, min, max, and standard deviation

## Expected Results

Based on the API design, you should see:

- **Payload size reduction**: ~33% smaller (no base64 encoding overhead)
- **Request time improvement**: 20-40% faster for large images (varies by image size)
- **Memory usage**: Lower peak memory usage during parsing

### Performance by Image Size

| Image Size | Expected Improvement |
|------------|---------------------|
| 100 KB     | ~15-25% faster      |
| 1 MB       | ~25-35% faster      |
| 5 MB       | ~35-45% faster      |
| 10 MB      | ~40-50% faster      |

Larger images benefit more from avoiding base64 encoding.

## Sample Output

```
Loading image: test.jpg
Image size: 2048.5 KB (2.00 MB)

============================================================
BENCHMARKING OLD API (base64 JSON)
============================================================
Converting image to base64...
Running old API benchmark (10 iterations)...
  Run 1/10: 1.234s (mem delta: 45.2MB)
  Run 2/10: 1.198s (mem delta: 43.8MB)
  ...

============================================================
BENCHMARKING NEW API (multipart)
============================================================
Running new API benchmark (10 iterations)...
  Run 1/10: 0.856s (mem delta: 28.1MB)
  Run 2/10: 0.843s (mem delta: 27.9MB)
  ...

============================================================
COMPARISON RESULTS
============================================================

Image size: 2048.5 KB
Model: my-project/1
Runs: 10

--- OLD API (base64 JSON) ---
  Base64 encoding time: 0.0342s
  Payload size: 2731.2 KB
  Mean time: 1.215s
  Median time: 1.210s
  Min time: 1.198s
  Max time: 1.234s
  Std dev: 0.012s

--- NEW API (multipart) ---
  Base64 encoding time: 0.0000s
  Payload size: 2049.0 KB
  Mean time: 0.851s
  Median time: 0.849s
  Min time: 0.843s
  Max time: 0.862s
  Std dev: 0.006s

--- IMPROVEMENTS ---
  Time improvement: +30.0%
  Payload size reduction: +25.0%
  Speedup factor: 1.43x
```

## Testing Different Scenarios

### Test with Small Images (Quick Tests)

```bash
# Generate a small test image
python -c "from PIL import Image; Image.new('RGB', (640, 480), color='red').save('small.jpg')"

python -m tests.benchmark.api_performance \
    --api-key YOUR_KEY \
    --model-id YOUR_MODEL \
    --image-path small.jpg \
    --runs 5
```

### Test with Large Images (Real-world Scenarios)

```bash
# Generate a large test image
python -c "from PIL import Image; Image.new('RGB', (4096, 3072), color='blue').save('large.jpg')"

python -m tests.benchmark.api_performance \
    --api-key YOUR_KEY \
    --model-id YOUR_MODEL \
    --image-path large.jpg \
    --runs 10
```

### High-Volume Testing

For simulating high-volume production scenarios:

```bash
python -m tests.benchmark.api_performance \
    --api-key YOUR_KEY \
    --model-id YOUR_MODEL \
    --image-path production_sample.jpg \
    --runs 100
```

## Troubleshooting

### Server Not Responding

Ensure your inference server is running:

```bash
curl http://localhost:9001/info
```

### Authentication Errors

- Old API: Check that API key works in request body
- New API: Check that API key works in Authorization header

### Model Not Found

Ensure the model is loaded on the server:

```bash
curl -X POST http://localhost:9001/model/add \
    -H "Content-Type: application/json" \
    -d '{"model_id": "your-model/1", "api_key": "YOUR_KEY"}'
```

## Notes

- The first request to each API may be slower due to model loading
- Run multiple iterations (`--runs`) to get stable results
- Test with images similar to your production workload
- Network latency affects both APIs equally, so improvements are from parsing/encoding
