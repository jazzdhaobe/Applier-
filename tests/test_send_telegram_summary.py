import json
import os
import unittest
from unittest.mock import patch

from scripts import send_telegram_summary


class SendTelegramSummaryTest(unittest.TestCase):
    def test_build_messages_keeps_each_message_under_telegram_limit(self):
        messages = send_telegram_summary.build_messages(
            ["A" * 2500, "B" * 2500]
        )

        self.assertEqual(len(messages), 2)
        self.assertTrue(
            all(
                len(message) <= send_telegram_summary.MAX_TELEGRAM_MESSAGE_LENGTH
                for message in messages
            )
        )

    def test_skips_when_bot_secrets_are_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(send_telegram_summary, "send_message") as send:
                self.assertEqual(send_telegram_summary.main(), 0)

        send.assert_not_called()

    def test_rejects_partially_configured_bot_secrets(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token"},
            clear=True,
        ):
            self.assertEqual(send_telegram_summary.main(), 1)

    def test_sends_titles_from_successful_application_log(self):
        log_text = (
            "✅ Applied to 1/30 jobs. Title='Backend Engineer'\n"
            "[SKIP] Job 2: Apply not confirmed\n"
            "✅ Applied to 2/30 jobs. Title='AI Developer'\n"
        )
        environment = {
            "TELEGRAM_BOT_TOKEN": "bot-token",
            "TELEGRAM_CHAT_ID": "123456789",
        }
        with patch.dict(os.environ, environment, clear=True):
            with patch.object(send_telegram_summary.Path, "exists", return_value=True):
                with patch.object(
                    send_telegram_summary.Path,
                    "read_text",
                    return_value=log_text,
                ):
                    with patch.object(
                        send_telegram_summary,
                        "send_message",
                    ) as send:
                        self.assertEqual(send_telegram_summary.main(), 0)

        send.assert_called_once_with(
            "bot-token",
            "123456789",
            "Applied jobs (1/1):\n- Backend Engineer\n- AI Developer",
        )

    def test_send_message_posts_telegram_request(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"ok": true, "result": {}}'

        with patch.object(
            send_telegram_summary,
            "urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            send_telegram_summary.send_message(
                "bot-token",
                "123456789",
                "Applied jobs:\n- Backend Engineer",
            )

        request = urlopen.call_args.args[0]
        self.assertIn("/botbot-token/sendMessage", request.full_url)
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "chat_id": "123456789",
                "text": "Applied jobs:\n- Backend Engineer",
            },
        )


if __name__ == "__main__":
    unittest.main()
