"""Send confirmed job application titles to a Telegram chat."""

import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_TELEGRAM_MESSAGE_LENGTH = 3900
MESSAGE_HEADER_LENGTH = len("Applied jobs (999/999):\n")


def build_messages(titles):
    chunks = []
    current_titles = []
    current_length = MESSAGE_HEADER_LENGTH

    for title in titles:
        line = f"- {title}"
        if current_titles and current_length + len(line) + 1 > MAX_TELEGRAM_MESSAGE_LENGTH:
            chunks.append(current_titles)
            current_titles = []
            current_length = MESSAGE_HEADER_LENGTH
        if len(line) + MESSAGE_HEADER_LENGTH > MAX_TELEGRAM_MESSAGE_LENGTH:
            line = line[: MAX_TELEGRAM_MESSAGE_LENGTH - MESSAGE_HEADER_LENGTH - 3] + "..."
        current_titles.append(line)
        current_length += len(line) + 1

    if current_titles:
        chunks.append(current_titles)

    total = len(chunks)
    return [
        f"Applied jobs ({index}/{total}):\n" + "\n".join(chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]


def send_message(bot_token, chat_id, message):
    endpoint = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"Telegram sendMessage failed with HTTP {response.status}."
                )
            result = json.loads(response.read().decode("utf-8"))
            if not result.get("ok"):
                raise RuntimeError("Telegram API rejected the message.")
    except HTTPError as exc:
        raise RuntimeError(
            f"Telegram sendMessage failed with HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Telegram: {exc.reason}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Telegram returned an invalid response.") from exc


def main():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token and not chat_id:
        print("[WARN] Telegram notification skipped; bot secrets are not configured.")
        return 0
    if not bot_token or not chat_id:
        print("[ERROR] Telegram notification requires both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return 1

    log_path = Path("logs/apply_run.log")
    if not log_path.exists():
        print(f"[ERROR] Cannot send Telegram summary; {log_path} was not generated.")
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
        print("[INFO] Telegram notification skipped; no successful applications found.")
        return 0

    messages = build_messages(titles)
    for message in messages:
        send_message(bot_token, chat_id, message)

    print(
        f"[INFO] Sent Telegram application summary "
        f"({len(titles)} job(s), {len(messages)} message(s))."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
