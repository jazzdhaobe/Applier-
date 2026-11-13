import os
import json
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs

from scripts import send_whatsapp_summary


class SendWhatsAppSummaryTest(unittest.TestCase):
    def test_build_messages_keeps_each_message_under_whatsapp_limit(self):
        messages = send_whatsapp_summary.build_messages(
            ["A" * 1000, "B" * 1000]
        )

        self.assertEqual(len(messages), 2)
        self.assertTrue(
            all(
                len(message) <= send_whatsapp_summary.MAX_WHATSAPP_MESSAGE_LENGTH
                for message in messages
            )
        )

    def test_skips_when_twilio_secrets_are_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(send_whatsapp_summary, "send_message") as send:
                self.assertEqual(send_whatsapp_summary.main(), 0)

        send.assert_not_called()

    def test_sends_titles_from_successful_application_log(self):
        log_text = (
            "✅ Applied to 1/30 jobs. Title='Backend Engineer'\n"
            "[SKIP] Job 2: Apply not confirmed\n"
            "✅ Applied to 2/30 jobs. Title='AI Developer'\n"
        )
        environment = {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "secret",
            "TWILIO_WHATSAPP_CONTENT_SID": "HX123",
            "WHATSAPP_FROM": "whatsapp:+14155238886",
            "WHATSAPP_TO": "whatsapp:+19995550123",
        }
        with patch.dict(os.environ, environment, clear=True):
            with patch.object(send_whatsapp_summary.Path, "exists", return_value=True):
                with patch.object(
                    send_whatsapp_summary.Path,
                    "read_text",
                    return_value=log_text,
                ):
                    with patch.object(
                        send_whatsapp_summary,
                        "send_message",
                    ) as send:
                        self.assertEqual(send_whatsapp_summary.main(), 0)

        send.assert_called_once_with(
            "AC123",
            "secret",
            "whatsapp:+14155238886",
            "whatsapp:+19995550123",
            "HX123",
            "Applied jobs (1/1):\n- Backend Engineer\n- AI Developer",
        )

    def test_send_message_posts_twilio_whatsapp_request(self):
        class FakeResponse:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with patch.object(
            send_whatsapp_summary,
            "urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            send_whatsapp_summary.send_message(
                "AC123",
                "secret",
                "whatsapp:+14155238886",
                "whatsapp:+19995550123",
                "HX123",
                "Applied jobs:\n- Backend Engineer",
            )

        request = urlopen.call_args.args[0]
        self.assertIn("/Accounts/AC123/Messages.json", request.full_url)
        body = parse_qs(request.data.decode("utf-8"))
        self.assertEqual(body["From"], ["whatsapp:+14155238886"])
        self.assertEqual(body["To"], ["whatsapp:+19995550123"])
        self.assertEqual(body["ContentSid"], ["HX123"])
        self.assertEqual(
            json.loads(body["ContentVariables"][0]),
            {"1": "Applied jobs:\n- Backend Engineer"},
        )


if __name__ == "__main__":
    unittest.main()
