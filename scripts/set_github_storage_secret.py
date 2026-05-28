"""Upload .auth/storage_state.json to GitHub Actions secret JOBAUTO_STORAGE_STATE."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "JayeshDhobe/Auto-Job-Applier"
SECRET_NAME = "JOBAUTO_STORAGE_STATE"
DEFAULT_STATE = Path(__file__).resolve().parents[1] / ".auth" / "storage_state.json"


def get_github_token() -> str:
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise RuntimeError("Could not read GitHub token from git credential helper")


def github_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    from nacl import encoding, public

    public_key = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    if not args.state_file.is_file():
        print(f"[ERROR] Missing {args.state_file}. Run scripts/export_storage_state.py first.", file=sys.stderr)
        return 1

    raw = args.state_file.read_text(encoding="utf-8")
    json.loads(raw)
    size = len(raw.encode("utf-8"))
    if size > 65536:
        print(f"[ERROR] Storage state is {size} bytes; GitHub secret limit is 64 KB.", file=sys.stderr)
        return 1

    token = get_github_token()
    owner, repo = args.repo.split("/", 1)
    key_url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/public-key"
    key_data = github_request("GET", key_url, token)
    encrypted = encrypt_secret(key_data["key"], raw)
    put_url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{SECRET_NAME}"
    github_request(
        "PUT",
        put_url,
        token,
        {
            "encrypted_value": encrypted,
            "key_id": key_data["key_id"],
        },
    )
    print(f"[OK] Updated {args.repo} secret {SECRET_NAME} ({size} bytes from {args.state_file})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
