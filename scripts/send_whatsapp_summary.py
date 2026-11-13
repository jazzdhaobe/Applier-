"""Send the applied-job titles from a workflow run through Twilio WhatsApp."""

import base64
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MAX_WHATSAPP_MESSAGE_LENGTH = 1600


def build_messages(titles):
    chunks = []
    current_titles = []
    current_length = len("Applied jobs (999/999):\n")

    for title in titles:
        line = f"- {title}"
        if current_titles and current_length + len(line) + 1 > MAX_WHATSAPP_MESSAGE_LENGTH:
            chunks.append(current_titles)
            current_titles = []
            current_length = len("Applied jobs (999/999):\n")
        if len(line) + len("Applied jobs (999/999):\n") > MAX_WHATSAPP_MESSAGE_LENGTH:
            line = (
                line[: MAX_WHATSAPP_MESSAGE_LENGTH - len("Applied jobs (999/999):\n") - 3]
                + "..."
            )
        current_titles.append(line)
        current_length += len(line) + 1

    if current_titles:
        chunks.append(current_titles)

    total = len(chunks)
    return [
        f"Applied jobs ({index}/{total}):\n" + "\n".join(chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]


def send_message(account_sid, auth_token, sender, recipient, content_sid, message):
    endpoint = (
        f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    )
    credentials = base64.b64encode(
        f"{account_sid}:{auth_token}".encode("utf-8")
    ).decode("ascii")
    request = Request(
        endpoint,
        data=urlencode(
            {
                "From": sender,
                "To": recipient,
                "ContentSid": content_sid,
                "ContentVariables": json.dumps({"1": message}),
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"Twilio WhatsApp request failed with HTTP {response.status}."
                )
    except HTTPError as exc:
        raise RuntimeError(
            f"Twilio WhatsApp request failed with HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Twilio: {exc.reason}") from exc


def main():
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    sender = os.environ.get("WHATSAPP_FROM", "").strip()
    recipient = os.environ.get("WHATSAPP_TO", "").strip()
    content_sid = os.environ.get("TWILIO_WHATSAPP_CONTENT_SID", "").strip()
    credentials = (account_sid, auth_token, sender, recipient, content_sid)

    if not any(credentials):
        print("[WARN] WhatsApp notification skipped; Twilio secrets are not configured.")
        return 0
    if not all(credentials):
        print(
            "[ERROR] WhatsApp notification requires TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN, WHATSAPP_FROM, WHATSAPP_TO, and "
            "TWILIO_WHATSAPP_CONTENT_SID."
        )
        return 1

    log_path = Path("logs/apply_run.log")
    if not log_path.exists():
        print(f"[ERROR] Cannot send WhatsApp summary; {log_path} was not generated.")
        return 1

    text = log_path.read_text(encoding="utf-8", errors="replace")
    titles = [
        title.strip()
        for title in re.findall(
            r"✅ Applied to \d+/\d+ jobs\. Title='([^']*)'", text
        )
        if title.strip()
    ]
    if not titles:
        print("[INFO] WhatsApp notification skipped; no successful applications found.")
        return 0

    messages = build_messages(titles)
    for message in messages:
        send_message(
            account_sid,
            auth_token,
            sender,
            recipient,
            content_sid,
            message,
        )

    print(
        f"[INFO] Sent WhatsApp application summary "
        f"({len(titles)} job(s), {len(messages)} message(s))."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
