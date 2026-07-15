$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $repoRoot "scripts\run_daily_apply.ps1"
$taskName = "Auto Job Applier"
$secretDir = Join-Path $env:APPDATA "jobautobot"
$secretPath = Join-Path $secretDir "credentials.xml"

if (-not (Test-Path $runScript)) {
    throw "Run script not found at $runScript"
}

$userDataPath = Join-Path $repoRoot "jab\data\user_data.json"
$email = $env:JOBAUTO_EMAIL
if (-not $email -and (Test-Path $userDataPath)) {
    $userData = Get-Content -Raw $userDataPath | ConvertFrom-Json
    $email = $userData.Email
}
if (-not $email) {
    $email = Read-Host "Enter the Naukri email"
}

$password = $env:JOBAUTO_PASSWORD
if (-not $password) {
    $securePassword = Read-Host "Enter the Naukri password" -AsSecureString
    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    try {
        $password = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) | Out-Null
    }
}

$search = $env:JOBAUTO_SEARCH
$location = $env:JOBAUTO_LOCATION
$experience = $env:JOBAUTO_EXPERIENCE
$jobAge = $env:JOBAUTO_JOBAGE

New-Item -ItemType Directory -Path $secretDir -Force | Out-Null
$securePassword = ConvertTo-SecureString -String $password -AsPlainText -Force
$securePassword | Export-Clixml -Path $secretPath

$env:JOBAUTO_EMAIL = $email
$env:JOBAUTO_SEARCH = $search
$env:JOBAUTO_LOCATION = $location
$env:JOBAUTO_EXPERIENCE = $experience
$env:JOBAUTO_JOBAGE = $jobAge

setx JOBAUTO_EMAIL $email | Out-Null
if ($search) { setx JOBAUTO_SEARCH $search | Out-Null }
if ($location) { setx JOBAUTO_LOCATION $location | Out-Null }
if ($experience) { setx JOBAUTO_EXPERIENCE $experience | Out-Null }
if ($jobAge) { setx JOBAUTO_JOBAGE $jobAge | Out-Null }

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`""
# Create 5 daily triggers (IST): 07:00, 09:00, 12:00, 15:00, 17:00
$trigger1 = New-ScheduledTaskTrigger -Daily -At 7am
$trigger2 = New-ScheduledTaskTrigger -Daily -At 9am
$trigger3 = New-ScheduledTaskTrigger -Daily -At 12pm
$trigger4 = New-ScheduledTaskTrigger -Daily -At 3pm
$trigger5 = New-ScheduledTaskTrigger -Daily -At 5pm
$triggers = @($trigger1, $trigger2, $trigger3, $trigger4, $trigger5)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RunOnlyIfNetworkAvailable -StartWhenAvailable

# Register the scheduled task for the current user (avoids requiring elevated privileges)
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings -Force | Out-Null


Write-Host "Daily task created successfully."
Write-Host "The task will run every day at 9:00 AM for this user."
