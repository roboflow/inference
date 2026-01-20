# File & Download Errors

**Base Classes:** `RetryError`, `UntrustedFileError`, `FileHashSumMissmatch`

File and download errors occur when the system fails to download model files or verify their integrity. These errors happen during the file download phase and are typically caused by network issues, corrupted downloads, or security validation failures.

---

## RetryError

**Transient network or server errors during download operations.**

### Overview

This error occurs when a network request fails due to connectivity issues or the server returns a retryable error code. The system uses automatic retry logic with exponential backoff, but if all retries are exhausted, this error is raised.

### When It Occurs

**Scenario 1: Network connectivity issues**

- Internet connection lost or unstable

- DNS resolution failures

- Firewall blocking requests

- Proxy configuration issues

**Scenario 2: Server returns retryable status codes**

- 5xx server errors (500, 502, 503, 504)

- 429 Too Many Requests

- Temporary server unavailability

**Scenario 3: Connection timeouts**

- Request takes longer than configured timeout

- Slow network connection

- Server not responding

**Scenario 4: Server doesn't support required features**

- Server doesn't support range requests (HTTP 206)

- Missing required HTTP headers

### What To Check

1. **Verify network connectivity:**
   ```bash
   # Test connection to Roboflow API
   curl -I https://api.roboflow.com

   # Test DNS resolution
   nslookup api.roboflow.com
   ```

2. **Check firewall/proxy settings:**

     * Ensure outbound HTTPS connections are allowed

     * Verify proxy configuration if applicable

     * Check corporate firewall rules

3. **Review error message:**

     * "Connectivity error" → Network issue
    
     * "returned response code" → Server error
    
     * "does not support range requests" → Server limitation

4. **Check Roboflow API status:**

     * Visit [Roboflow Status Page](https://status.roboflow.com)

     * Check for ongoing incidents

### How To Fix

**Network connectivity issues:**

```bash
# Test basic connectivity
ping api.roboflow.com

# Check if you can reach the API
curl https://api.roboflow.com

# If behind proxy, set environment variables
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

**Increase timeout values:**

```python
import os

# Increase API timeout (default is usually 30 seconds)
os.environ["API_CALLS_TIMEOUT"] = "120"  # 2 minutes

from inference_models import AutoModel
model = AutoModel.from_pretrained("yolov8n-640")
```

**Wait and retry:**

- The error is often temporary
- Wait a few minutes and try again
- The system already retries automatically, so if you see this error, retries were exhausted

**Check server status:**

- If Roboflow API is down, wait for service restoration
- Check [status.roboflow.com](https://status.roboflow.com) for updates


---

## UntrustedFileError

**File lacks required hash sum for verification.**

### Overview

This error occurs when attempting to download a file that doesn't have an MD5 hash sum for integrity verification. This is a security measure to prevent downloading potentially corrupted or tampered files.

### When It Occurs

**Scenario 1: Missing hash sum in metadata**

- Roboflow API doesn't provide MD5 hash for file

- Model package metadata incomplete

- Old model version without hash sums

**Scenario 2: Strict security mode enabled**

- `download_files_without_hash=False` is set in `AutoModel.from_pretrained(...)`

- Security policy requires hash verification

- Corporate security requirements

**Scenario 3: Misconfigured download procedure**

- Attempting to name file by hash when hash is missing

- Internal code bug in download logic

### What To Check

1. **Check if this is expected:**

     * Some older models may not have hash sums

     * Check model metadata for hash availability

2. **Review security requirements:**

     * Determine if hash verification is required for your use case

     * Check organizational security policies


### How To Fix

**This is typically a Roboflow API issue:**

If you're using Roboflow models and see this error:

- [Report the issue](https://github.com/roboflow/inference/issues) with:

  - Model ID

  - Model version

  - Full error message

**For custom weights providers:**

Ensure your provider includes MD5 hash sums in file metadata:

```python
# use for debugging only - this is not part of Public API
from inference_models.weights_providers.entities import FileDownloadSpecs

# ✅ Correct - include MD5 hash
file_spec = FileDownloadSpecs(
    file_handle="model.onnx",
    download_url="https://example.com/model.onnx",
    md5_hash="abc123def456..."  # Include this!
)

# ❌ Wrong - missing hash
file_spec = FileDownloadSpecs(
    file_handle="model.onnx",
    download_url="https://example.com/model.onnx",
    md5_hash=None  # Will cause UntrustedFileError
)
```

---

## FileHashSumMissmatch

**Downloaded file hash doesn't match the expected value.**

### Overview

This error occurs when a file is successfully downloaded, but its computed MD5 hash doesn't match the expected hash provided in the metadata. This indicates the file may be corrupted or tampered with.

### When It Occurs

**Scenario 1: Corrupted download**

- Network issues during download

- Partial file transfer

- Data corruption in transit

**Scenario 2: File modified on server**

- File changed on server after metadata was generated

- CDN serving stale or wrong version

- Cache inconsistency

**Scenario 3: Disk write errors**

- Disk full during download

- Filesystem errors

- Permission issues

**Scenario 4: Metadata inconsistency**

- Wrong hash in metadata

- Hash for different file version

- Provider metadata bug

### What To Check

1. **Check available disk space:**
   ```bash
   df -h
   ```

2. **Verify network stability:**

     * Check for packet loss

     * Test download speed

     * Look for network interruptions

3. **Check error message:**

     * Shows expected vs. calculated hash

     * Indicates which file failed

### How To Fix

**Clear cache and retry:**

```python
from inference_models.configuration import INFERENCE_HOME
import shutil
from pathlib import Path

# Clear model cache
cache_dir = Path(INFERENCE_HOME)
if cache_dir.exists():
    shutil.rmtree(cache_dir)

# Try downloading again
from inference_models import AutoModel
model = AutoModel.from_pretrained("yolov8n-640")
```

**Or use command line:**

```bash
# Clear cache directory - default location
rm -rf /tmp/cache

# Or if using custom cache location
rm -rf $MODEL_CACHE_DIR
```

**Check disk space:**

```bash
# Free up disk space if needed
df -h

# Clean up old files
# (be careful with this command)
```

**Verify network stability:**

- Use wired connection instead of WiFi if possible
- Disable VPN temporarily to test
- Try from different network

**If problem persists:**

This may indicate a provider issue:

- [Report the issue](https://github.com/roboflow/inference/issues) with:

  - Model ID

  - Expected hash from error message

  - Calculated hash from error message

  - When the error started occurring

**For custom weights providers:**

Ensure hash sums in metadata are correct:

```python
import hashlib

# Calculate correct MD5 hash
def calculate_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

# Use this hash in your metadata
correct_hash = calculate_md5("model.onnx")
```

