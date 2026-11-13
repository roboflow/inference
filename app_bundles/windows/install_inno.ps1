# Inno Setup ships preinstalled on windows-2022 runner images; install via
# Chocolatey only if missing (jrsoftware.org no longer serves direct downloads).
$innoPath = "C:\Program Files (x86)\Inno Setup 6"
if (-not (Test-Path (Join-Path $innoPath "ISCC.exe"))) {
  echo "Inno Setup not found, installing via Chocolatey..."
  choco install innosetup -y --no-progress
  if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install Inno Setup via Chocolatey"; exit 1 }
}
if (-not (Test-Path (Join-Path $innoPath "ISCC.exe"))) {
  Write-Error "ISCC.exe not found at $innoPath after installation."
  exit 1
}
echo "Adding Inno Setup directory to PATH: $innoPath"
Add-Content -Path $env:GITHUB_PATH -Value $innoPath
echo "Inno Setup available and added to PATH."
