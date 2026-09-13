$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 or newer is required.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\python.exe' scripts\fetch_malecns.py
if ($LASTEXITCODE -ne 0) { throw 'Dataset download or integrity verification failed.' }
& '.\.venv\Scripts\python.exe' scripts\prepare_connectome.py
if ($LASTEXITCODE -ne 0) { throw 'Dataset compilation failed.' }
Write-Output 'Ready. Run .\start.ps1 -OpenBrowser'
