param([int]$Port = 8877, [switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskExecutable = Join-Path $taskRoot 'Neurons.exe'
if (-not (Test-Path -LiteralPath $taskExecutable) -and -not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 first, or use the Windows release package.' }
$taskLogs = Join-Path $taskRoot 'logs'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $taskRoot '.runtime') -Force | Out-Null
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = Join-Path $taskRoot '.runtime\models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$taskOllama = Join-Path $taskRoot '.runtime\ollama\ollama.exe'
try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/tags' -TimeoutSec 2 }
catch {
    if (Test-Path -LiteralPath $taskOllama) {
        $taskProcess = Start-Process -FilePath $taskOllama -ArgumentList 'serve' -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogs 'ollama-out.log') -RedirectStandardError (Join-Path $taskLogs 'ollama-err.log') -PassThru
        $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRoot '.runtime\ollama.pid')
    }
}
$taskReady = $false
try {
    $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
    $taskReady = $taskHealth.app -eq 'Neurons'
} catch { }
if (-not $taskReady) {
    $taskProbe = New-Object System.Net.Sockets.TcpClient
    try {
        $taskProbe.Connect('127.0.0.1', $Port)
        throw "Port $Port is already in use by another service. Choose a different -Port."
    } catch [System.Net.Sockets.SocketException] { }
    finally { $taskProbe.Dispose() }
    if (Test-Path -LiteralPath $taskExecutable) {
        $taskProcess = Start-Process -FilePath $taskExecutable -ArgumentList @('--port',"$Port") -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogs 'app-out.log') -RedirectStandardError (Join-Path $taskLogs 'app-err.log') -PassThru
    } else {
        $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('app.py','--port',"$Port") -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogs 'app-out.log') -RedirectStandardError (Join-Path $taskLogs 'app-err.log') -PassThru
    }
    $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRoot '.runtime\app.pid')
    $taskDeadline = (Get-Date).AddSeconds(45)
    while (-not $taskReady -and (Get-Date) -lt $taskDeadline) {
        $taskProcess.Refresh()
        if ($taskProcess.HasExited) { throw "Neurons stopped during startup. See $taskLogs\app-err.log." }
        try {
            $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
            $taskReady = $taskHealth.app -eq 'Neurons' -and $taskHealth.ready
        } catch { }
        if (-not $taskReady) { Start-Sleep -Milliseconds 300 }
    }
    if (-not $taskReady) { throw "Startup did not become ready within 45 seconds. See $taskLogs\app-out.log." }
}
Write-Output "Neurons: http://127.0.0.1:$Port"
if ($OpenBrowser) { Start-Process "http://127.0.0.1:$Port" }
