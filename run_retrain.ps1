$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonCmd = "python"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logDir = Join-Path $projectRoot "logs"
$logFile = Join-Path $logDir "retrain_$timestamp.log"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Write-Output "[$(Get-Date)] Starting retraining..." | Tee-Object -FilePath $logFile -Append

& $pythonCmd "fakenewsprediction.py" --data "compressed_data.csv" --model-out "model.joblib" --vectorizer-out "vectorizer.joblib" 2>&1 `
    | Tee-Object -FilePath $logFile -Append

Write-Output "[$(Get-Date)] Retraining completed." | Tee-Object -FilePath $logFile -Append
