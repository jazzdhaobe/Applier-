$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $repoRoot "logs"
$secretPath = Join-Path $env:APPDATA "jobautobot\credentials.xml"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path $logDir "daily-apply-$timestamp.log"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp $Message" | Tee-Object -FilePath $logFile -Append | Out-Null
}

if (-not (Test-Path $pythonPath)) {
    throw "Python executable not found at $pythonPath"
}

if (-not (Test-Path $secretPath)) {
    throw "Secure password file not found at $secretPath. Run scripts/setup_daily_task.ps1 first."
}

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$requiredEnv = @("JOBAUTO_EMAIL")
foreach ($name in $requiredEnv) {
    if (-not (Get-Item env:$name -ErrorAction SilentlyContinue)) {
        throw "Environment variable $name is required. Run scripts/setup_daily_task.ps1 first."
    }
}

$securePassword = Import-Clixml -Path $secretPath
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) | Out-Null
}

$env:JOBAUTO_PASSWORD = $plainPassword
$env:PYTHONUNBUFFERED = "1"

Write-Log "Starting Auto-Job-Applier run"
Write-Log "Using email: $env:JOBAUTO_EMAIL"

try {
    & $pythonPath -m playwright install chromium 2>&1 | Tee-Object -FilePath $logFile -Append | Out-Null
    & $pythonPath -m jab --email $env:JOBAUTO_EMAIL --apply --filters --headless 2>&1 | Tee-Object -FilePath $logFile -Append

    if ($LASTEXITCODE -ne 0) {
        throw "python -m jab failed with exit code $LASTEXITCODE"
    }

    Write-Log "Auto-Job-Applier run completed successfully"
    exit 0
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    exit 1
}
