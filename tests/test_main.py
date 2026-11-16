import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

import jab.__main__ as main_module


class VerifyLoginCLIBehaviorTest(unittest.TestCase):
    def test_verify_login_runs_without_apply_flag(self):
        calls = {}

        class DummyBot:
            def __init__(self, *args, **kwargs):
                calls["init"] = True

            def init_browser(self):
                calls["init_browser"] = True

            def login(self):
                calls["login"] = True
                return True

            def close(self):
                calls["close"] = True

        with patch.object(main_module, "NaukriBot", DummyBot):
            with patch.object(sys, "argv", ["jab", "--email", "user@example.com", "--verify-login"]):
                with self.assertRaises(SystemExit) as exc:
                    main_module.main()

        self.assertEqual(exc.exception.code, 0)
        self.assertTrue(calls.get("init_browser"))
        self.assertTrue(calls.get("login"))
        self.assertTrue(calls.get("close"))

    def test_default_search_and_experience_are_focused(self):
        args = Namespace(
            password="test-password",
            storage_state=None,
            fresh_login=True,
            save_storage_state=None,
            email="user@example.com",
            number=None,
            search=None,
            experience=None,
            location=None,
            job_age=None,
            headless=False,
            otp=None,
        )

        with patch.object(main_module, "get_env", side_effect=lambda name, default=None: default):
            config = main_module.resolve_config(args)

        self.assertEqual(config["experience"], 3)
        self.assertEqual(config["search"], main_module.DEFAULT_SEARCH)
        self.assertNotIn("LangGraph", config["search"])


if __name__ == "__main__":
    unittest.main()
