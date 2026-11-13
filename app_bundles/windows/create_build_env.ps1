# Runner-wide packages (such as pipx) are not application dependencies. Keep
# wheel preparation, final resolution and freezing in a clean Python environment.
$buildEnvironment = Join-Path $env:RUNNER_TEMP "inference-windows-build"
python -m venv $buildEnvironment
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $buildEnvironment "Scripts")
Add-Content -Path $env:GITHUB_ENV -Value "VIRTUAL_ENV=$buildEnvironment"
