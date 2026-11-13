import unittest
from unittest.mock import MagicMock, patch

from jab.modules.naukri import NaukriBot


class NaukriNavigationTest(unittest.TestCase):
    def test_apply_resolves_relative_job_links(self):
        bot = object.__new__(NaukriBot)
        bot.page = MagicMock()
        bot.page.url = "https://www.naukri.com/gen-ai-jobs?k=ai"
        bot.page.locator.side_effect = RuntimeError("job card title unavailable")
        bot.applied_count = 0
        bot.applno = 1
        bot.page_no = 15

        with patch.object(
            bot,
            "_collect_naukri_apply_job_links",
            return_value=["/job-listings-data-scientist-hero-fincorp"],
        ), patch.object(bot, "_job_already_applied", return_value=True):
            bot.apply_()

        bot.page.goto.assert_called_once_with(
            "https://www.naukri.com/job-listings-data-scientist-hero-fincorp",
            timeout=30000,
        )

    def test_invalid_job_url_restarts_search_from_home_once(self):
        bot = object.__new__(NaukriBot)
        bot.page = MagicMock()
        bot.page.url = "https://www.naukri.com/gen-ai-jobs?k=ai"
        bot.page_no = 8
        bot._navigation_recovery_attempted = False
        bot._ensure_authenticated_session = MagicMock(return_value=True)
        bot.filter_ = MagicMock()

        self.assertTrue(bot._restart_search_after_invalid_job_url())
        self.assertFalse(bot._restart_search_after_invalid_job_url())

        bot.page.goto.assert_called_once_with(
            "https://www.naukri.com/mnjuser/homepage",
            timeout=40000,
        )
        bot.filter_.assert_called_once_with()
        self.assertEqual(bot.page_no, 1)
        self.assertEqual(bot.base_page_url, "https://www.naukri.com/gen-ai-jobs?k=ai")

    def test_apply_restarts_after_invalid_job_navigation(self):
        bot = object.__new__(NaukriBot)
        bot.page = MagicMock()
        bot.page.url = "https://www.naukri.com/gen-ai-jobs?k=ai"
        bot.page.goto.side_effect = RuntimeError("Cannot navigate to invalid URL")
        bot.page.locator.side_effect = RuntimeError("job card title unavailable")
        bot.applied_count = 0
        bot.applno = 1
        bot.page_no = 1

        def restart_search():
            bot.page_no = 15
            return True

        with patch.object(
            bot,
            "_collect_naukri_apply_job_links",
            side_effect=[
                ["/job-listings-data-scientist-hero-fincorp"],
                [],
            ],
        ), patch.object(
            bot,
            "_restart_search_after_invalid_job_url",
            side_effect=restart_search,
        ) as restart:
            bot.apply_()

        restart.assert_called_once_with()
        bot.page.goto.assert_called_once_with(
            "https://www.naukri.com/job-listings-data-scientist-hero-fincorp",
            timeout=30000,
        )


if __name__ == "__main__":
    unittest.main()
