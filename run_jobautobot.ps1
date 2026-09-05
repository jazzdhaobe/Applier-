$ErrorActionPreference = "Continue"

$repo = "C:\Users\Jayesh Dhobe\OneDrive\Desktop\jobautobot"
$python = "$repo\.venv\Scripts\python.exe"
$logDir = "$repo\logs"

# Make sure the log directory exists
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Force Python to see the project root
$env:PYTHONPATH = $repo

# Move into the project directory
Set-Location -LiteralPath $repo

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile = "$logDir\jobautobot_$timestamp.log"

"========================================" | Out-File $logFile
"JobAutoBot started: $(Get-Date)" | Out-File $logFile -Append
"Working directory: $(Get-Location)" | Out-File $logFile -Append
"Python: $python" | Out-File $logFile -Append
"PYTHONPATH: $env:PYTHONPATH" | Out-File $logFile -Append
"========================================" | Out-File $logFile -Append

# Verify that Python can find jab
& $python -c "import sys; print('Python:', sys.executable); import jab; print('JAB:', jab.__file__)" *>&1 |
    Tee-Object -FilePath $logFile -Append

"Running JobAutoBot..." | Out-File $logFile -Append

& $python -m jab --email jayeshdhobe7@gmail.com --apply --filters *>&1 |
    Tee-Object -FilePath $logFile -Append

$exitCode = $LASTEXITCODE

"" | Out-File $logFile -Append
"========================================" | Out-File $logFile -Append
"Exit Code: $exitCode" | Out-File $logFile -Append
"Finished: $(Get-Date)" | Out-File $logFile -Append
"========================================" | Out-File $logFile -Append

exit $exitCode