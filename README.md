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

### Run in the cloud (recommended if your laptop is off)

You can run the applier from GitHub Actions so it runs every day even if your laptop is closed.

- Add repository secrets (Settings → Secrets → Actions):
  - `JOBAUTO_EMAIL` — your Naukri email
  - `JOBAUTO_PASSWORD` — your Naukri password
  - `JOBAUTO_STORAGE_STATE` — full JSON from `.auth/storage_state.json` (required so CI can reuse your session and skip OTP)
- The workflow is at `.github/workflows/daily_apply.yml` and runs daily at 09:00 IST (50 applications with `--filters`). It can also be triggered manually via the Actions tab.

**Export / refresh session for CI** (run locally after logging in once if OTP appears):

```powershell
python scripts/export_storage_state.py --email your-email@gmail.com --password your-password
python scripts/set_github_storage_secret.py
```

Or paste the contents of `.auth/storage_state.json` into the `JOBAUTO_STORAGE_STATE` secret manually.

Note: Naukri often asks for OTP on password login from cloud IPs. Your home PC may not ask OTP, but GitHub Actions will unless `JOBAUTO_STORAGE_STATE` is set. Re-export the session every few weeks if CI login starts failing.

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
