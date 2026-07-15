## TODO - Ensure search-before-apply + improve verification

- [ ] Fix `start_apply()` / `bot_actions()` so it always performs keyword search + filters after login (use existing config: self.search/self.experience_years/self.location/self.jobage).
- [ ] Add logging of: keyword/search/location/experience + landing URL after search, and for each applied job the job URL/title (or screenshot + URL) so we can confirm actions happened.

- [ ] Add strong logging: current search results URL, keyword used, and list of applied job URLs/titles.
- [ ] Add a “proof of apply” counter increment only after confirmation (tighten `_complete_apply_after_click`).
- [ ] Update workflow/scheduler command if needed so it passes correct flags.
- [ ] Test locally with a dry run + short apply run; validate logs.

