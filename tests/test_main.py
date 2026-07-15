import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
