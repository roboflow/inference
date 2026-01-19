# File & Download Errors

**Base Classes:** `RetryError`, `UntrustedFileError`, `FileHashSumMissmatch`

File and download errors occur when the system fails to download model files or verify their integrity. These errors happen during the file download phase and are typically caused by network issues, corrupted downloads, or security validation failures.

## Common Characteristics

- **When they occur:** During model file downloads and integrity verification
- **Root causes:** Network failures, corrupted downloads, missing hash sums, hash mismatches
- **Impact:** Cannot download or verify model files; model loading cannot proceed
- **Recovery:** Retry download, check network connectivity, clear cache, verify file integrity

## Error Hierarchy

```
BaseInferenceError
├── RetryError
├── FileHashSumMissmatch
└── UntrustedFileError
```

## Error Types

### RetryError

**Transient network or server errors during download operations.**

### FileHashSumMissmatch

**Downloaded file hash doesn't match the expected value.**

### UntrustedFileError

**File lacks required hash sum for verification.**

