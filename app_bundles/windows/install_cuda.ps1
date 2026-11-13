$version = "12.6.3"
$short   = "12.6"
$url     = "https://developer.download.nvidia.com/compute/cuda/$version/network_installers/cuda_${version}_windows_network.exe"

Write-Host "Downloading CUDA $version network installer..."
Invoke-WebRequest -Uri $url -OutFile cuda_installer.exe -UseBasicParsing

Write-Host "Installing sub-packages..."
$proc = Start-Process -FilePath .\cuda_installer.exe -ArgumentList @(
  "-s",
  "nvcc_$short",
  "cudart_$short",
  "cublas_$short",
  "cublas_dev_$short",
  "curand_$short",
  "curand_dev_$short",
  "cuda_profiler_api_$short",
  "visual_studio_integration_$short"
) -Wait -NoNewWindow -PassThru

if ($proc.ExitCode -ne 0) {
  Write-Error "CUDA installer failed with exit code $($proc.ExitCode)"
  exit $proc.ExitCode
}

$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$short"
Write-Host "CUDA installed at: $cudaPath"

Write-Host "Contents of CUDA path:"
Get-ChildItem $cudaPath -ErrorAction SilentlyContinue
Write-Host "Contents of include dir:"
Get-ChildItem "$cudaPath\include" -ErrorAction SilentlyContinue | Select-Object -First 10

if (Test-Path "$cudaPath\include\cuda.h") {
  Write-Host "cuda.h found"
} else {
  Write-Host "ERROR: cuda.h NOT found"
  exit 1
}

# Replicate the outputs that Jimver/cuda-toolkit provides so downstream
# steps using steps.cuda-toolkit.outputs.CUDA_PATH require no changes.
"CUDA_PATH=$cudaPath" | Out-File -Append $env:GITHUB_OUTPUT
"CUDA_PATH=$cudaPath" | Out-File -Append $env:GITHUB_ENV
"CUDA_HOME=$cudaPath" | Out-File -Append $env:GITHUB_ENV
"INCLUDE=$cudaPath\include;$env:INCLUDE"     | Out-File -Append $env:GITHUB_ENV
"LIB=$cudaPath\lib\x64;$env:LIB"             | Out-File -Append $env:GITHUB_ENV
"$cudaPath\bin"                              | Out-File -Append $env:GITHUB_PATH
