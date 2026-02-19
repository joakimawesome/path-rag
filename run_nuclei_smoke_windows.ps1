param(
    [string]$InputDir = "",
    [string]$OutputRoot = "",
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($InputDir)) {
    $InputDir = if ($env:PATHRAG_INPUT_DIR) { $env:PATHRAG_INPUT_DIR } else { "C:\\path-rag-data\\wsi_raw" }
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = if ($env:PATHRAG_OUTPUT_ROOT) { $env:PATHRAG_OUTPUT_ROOT } else { "C:\\path-rag-data\\wsi_processed" }
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = if ($env:PATHRAG_PYTHON) { $env:PATHRAG_PYTHON } else { "python" }
}

$repoRoot = Split-Path -Parent $PSCommandPath
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tempRoot = if ($env:TEMP) { $env:TEMP } elseif ($env:TMP) { $env:TMP } else { [System.IO.Path]::GetTempPath() }
$smokeOutput = Join-Path $OutputRoot "smoke_$timestamp"
$tempDir = Join-Path $tempRoot "nuclei_smoke_$timestamp"
$smokeList = Join-Path $tempDir "smoke_first_wsi.txt"

if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "Input directory does not exist: $InputDir"
}

New-Item -ItemType Directory -Path $smokeOutput -Force | Out-Null
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

$firstWsi = Get-ChildItem -Path $InputDir -Filter *.svs -File | Sort-Object Name | Select-Object -First 1
if (-not $firstWsi) {
    throw "No .svs files found under $InputDir"
}

$firstWsi.FullName | Set-Content -Path $smokeList -Encoding UTF8

Push-Location $repoRoot
try {
    & $PythonExe process_wsi_batch.py `
        --input_dir $InputDir `
        --output_dir $smokeOutput `
        --temp_dir $tempDir `
        --file_list $smokeList
}
finally {
    Pop-Location
}

Write-Host "Smoke test complete. Output: $smokeOutput"
