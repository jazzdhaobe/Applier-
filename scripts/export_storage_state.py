"""Export Playwright storage state for GitHub Actions (JOBAUTO_STORAGE_STATE secret).

Run after a successful local login:

  python -m jab --email you@example.com --password ... ^
    --storage-state .auth/storage_state.json --save-storage-state .auth/storage_state.json

Or use this helper (opens a visible browser, logs in, writes .auth/storage_state.json):

  python scripts/export_storage_state.py --email you@example.com --password ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / ".auth" / "storage_state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Naukri session for CI")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--otp", help="OTP if Naukri prompts during export")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write storage_state.json (default: .auth/storage_state.json)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from jab.modules.naukri import NaukriBot

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    bot = NaukriBot(
        args.email,
        args.password,
        args.email,
        number=1,
        headless=False,
        otp=args.otp,
        storage_state_path=str(output) if output.exists() else None,
        save_storage_state_path=str(output),
    )
    bot.init_browser()
    try:
        if not bot.login():
            print("[ERROR] Login failed; could not export storage state.", file=sys.stderr)
            return 1
        bot.save_storage_state()
    finally:
        bot.close()

    data = json.loads(output.read_text(encoding="utf-8"))
    cookies = len(data.get("cookies", []))
    origins = len(data.get("origins", []))
    size = output.stat().st_size
    print(f"[OK] Wrote {output} ({size} bytes, cookies={cookies}, origins={origins})")
    print()
    print("Add to GitHub: Repository → Settings → Secrets → Actions → New secret")
    print("  Name:  JOBAUTO_STORAGE_STATE")
    print("  Value: entire contents of the file above (raw JSON, one line is fine)")
    print()
    print("Re-export and update the secret if CI starts asking for OTP again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
