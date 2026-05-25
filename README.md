# Auto-Job-Applier

Auto-Job-Applier automates Naukri job applications with Playwright and a local chatbot model. The current setup is designed to run daily at 9:00 AM through Windows Task Scheduler.

## Features

- Daily non-interactive application flow
- Headless browser execution for automation
- Configurable search keywords, experience, location, and job age
- Persisted training data and model under `jab/data/`
- Scheduled runner and setup scripts for Windows

## Setup

1. Create and activate the virtual environment.
2. Install dependencies.
3. Install the Chromium browser binary.
4. Train the model once.
5. Run the scheduler setup script.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
python -m jab --email your-email@gmail.com --train
powershell -ExecutionPolicy Bypass -File .\scripts\setup_daily_task.ps1
```

## Manual run

```powershell
.\.venv\Scripts\Activate.ps1
python -m jab --email your-email@gmail.com --apply --headless
```

## Daily automation

The repository includes:

- `scripts/setup_daily_task.ps1` to create a Windows Task Scheduler task at 9:00 AM
- `scripts/run_daily_apply.ps1` to launch the application and write logs under `logs/`

The task uses a secure credential file in `%APPDATA%\jobautobot\credentials.xml` and a persistent email/search configuration in Windows environment variables.

## Validation

You can validate the launcher without opening the browser:

```powershell
.\.venv\Scripts\Activate.ps1
python -m jab --email your-email@gmail.com --dry-run
```

## Notes

- The scheduler script will re-install Chromium if needed.
- Logs for each run are stored in the `logs/` directory and are ignored by git.
- The current defaults target 50 applications per run.
