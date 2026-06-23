import os
import time
import re
from pathlib import Path
import numpy as np
import nltk
import json
from urllib.parse import urlparse, urlunparse
from playwright.sync_api import sync_playwright , expect
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

CHROMIUM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def log_info(message: str) -> None:
    print(message, flush=True)


class ChatbotModel():
    def __init__(self, user_data):
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_words = set(['?', '!', '.', ','])
        self.user_data = user_data
        self.words = []
        self.words_set = set()
        self.classes = []
        self.load_data()
        self.load_model()

    def load_data(self):
        words_temp = []
        classes_set = set()
        for intent in self.user_data:
            classes_set.add(intent['tag'])
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                words_temp.extend(word_list)
        self.words = sorted(set([self.lemmatizer.lemmatize(w.lower()) for w in words_temp if w not in self.ignore_words]))
        self.words_set = set(self.words)
        self.classes = sorted(classes_set)
    
    def load_model(self):
        model_path = f"./jab/data/{user}/model.keras"
        self.model = tf.keras.models.load_model(model_path)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        sentence_set = set(sentence_words)
        bag = [1 if w in sentence_set else 0 for w in self.words]
        if show_details:
            for w in self.words:
                if w in sentence_set:
                    print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, ints):
        tag = ints[0]['intent']
        for intent in self.user_data:
            if intent['tag'] == tag:
                result = intent['answer']
                break
        res = lambda y: y if y not in " " else "A"
        return res(result)

    def chatbot_response(self, msg):
        try:
            ints = self.predict_class(msg)
            res = self.get_response(ints)
        except:
            res = 'A'
        return res

class ChatbotAgent:
    def __init__(self,page,username):
        global user
        user = username
        self.page = page
        with open(f"./jab/data/{user}/training_data.json", 'r') as json_file:
            user_data = json.load(json_file)    
        self.model = ChatbotModel(user_data)
        self.analyzer = SentimentIntensityAnalyzer()

    def sentiment_score(self,text):
        score = self.analyzer.polarity_scores(text)
        return score['compound'] 
    
    def match_by_sentiment(self,target, strings):
        target_sentiment = self.sentiment_score(target)
        best_match = min(
            ((s, abs(target_sentiment - self.sentiment_score(s))) for s in strings),
            key=lambda x: x[1]
        )
        return best_match

    def classify_new_question(self):
        page=self.page
        cbcn = page.wait_for_selector(".chatbot_MessageContainer",timeout = 3000)
        time.sleep(2)
        try:
            while cbcn:
                question_element = page.locator(".botMsg").last
                self.page.wait_for_timeout(1000)
                
                question = question_element.inner_text()
                
                # Check if this is an experience-related question
                experience_keywords = ['experience', 'years', 'yrs', 'how many years', 'total experience', 'work experience']
                is_experience_question = any(keyword.lower() in question.lower() for keyword in experience_keywords)
                
                if is_experience_question:
                    answer = "3"
                else:
                    answer = self.model.chatbot_response(question)

                checkboxes = cbcn.query_selector_all('input[type="checkbox"]')
                radio_buttons = cbcn.query_selector_all('input[type="radio"]')
                text_input = page.locator('.chatbot_MessageContainer .textArea')
                chip = page.query_selector('.chatbot_MessageContainer .chipsContainer .chatbot_Chip')
                suggs = cbcn.query_selector_all('.ssc__heading')
                dob = cbcn.query_selector(".dob__container")
                if chip:
                    chip.click()
                    continue
                elif radio_buttons or checkboxes:
                    _buttons = radio_buttons or checkboxes
                    options = [el.evaluate('el => el.id') for el in _buttons]
                    finnas, _ = self.match_by_sentiment(answer,options)
                    label_ = page.locator(f'label[for="{finnas}"]')
                    label_.click(force=True)
                elif text_input.is_visible():
                    text_input.type(answer,delay=100)
                elif suggs:
                    options = [el.evaluate('el => el.innerText') for el in suggs]
                    finnas, _ = self.match_by_sentiment(answer,options)
                    page.click(f'text="{finnas}"')
                elif dob:
                    dob = answer.strip().split("/")
                    page.locator("input[name='day']").type(dob[0],delay=100)
                    page.locator("input[name='month']").type(dob[1],delay=100)
                    page.locator("input[name='year']").type(dob[2],delay=100)

                else:
                    return 
                send = page.locator('.sendMsg')
                try:
                    expect(send).to_be_enabled()
                    time.sleep(0.5)
                    send.click(timeout=3000)
                except:
                    return
                time.sleep(1)
        except Exception as e:
            return {"response":'error occured on classify_new_question',"error":str(e)}

class NaukriBot:
    def __init__(
        self,
        usreml,
        usrpas,
        username,
        number=10,
        headless=False,
        otp=None,
        storage_state_path=None,
        save_storage_state_path=None,
    ):
        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None
        self.usr = [usreml, usrpas]
        self.username = username
        self.applno = number
        self.applied_count = 0
        self.page_no = 1
        self.tabs = ["profile", "apply", "preference", "similar_jobs"]
        self.pattern = re.compile(r'https://.*/myapply/saveApply\?strJobsarr=')
        self.headless = headless
        self.otp = otp
        self.storage_state_path = storage_state_path
        self.save_storage_state_path = save_storage_state_path or storage_state_path
        self.experience_years = "2"
        self.success_url_patterns = [
            re.compile(r'https://.*/myapply/saveApply\?strJobsarr='),
            re.compile(r'https://.*/myapply/thankyou.*'),
            re.compile(r'https://.*/apply-workflow.*'),
        ]

    def _chatbot_contexts(self):
        contexts = [self.page]
        for frame in self.page.frames:
            if frame != self.page.main_frame:
                contexts.append(frame)
        return contexts

    def _all_pages(self):
        return [self.page] + [p for p in self.context.pages if p != self.page]

    def _all_page_urls(self):
        return [p.url for p in self._all_pages()]

    def _search_contexts(self):
        contexts = []
        for page in self._all_pages():
            contexts.append(page)
            for frame in page.frames:
                if frame != page.main_frame:
                    contexts.append(frame)
        return contexts

    def _any_page_matches_success_url(self):
        for page in self._all_pages():
            for pattern in self.success_url_patterns:
                try:
                    if pattern.search(page.url):
                        return True
                except Exception:
                    continue
        return False

    def _answer_chatbot_heuristic(self):
        """Answer screening questions without the ML model (reliable in CI)."""
        for _ in range(25):
            handled = False
            for ctx in self._chatbot_contexts():
                container = ctx.locator(".chatbot_MessageContainer")
                try:
                    if not container.is_visible(timeout=400):
                        continue
                except Exception:
                    continue
                handled = True
                question = ""
                try:
                    question = ctx.locator(".botMsg").last.inner_text(timeout=2000)
                except Exception:
                    pass
                exp_keywords = (
                    "experience", "years", "yrs", "how many years",
                    "total experience", "work experience",
                )
                if any(keyword in question.lower() for keyword in exp_keywords):
                    answer = str(self.experience_years)
                else:
                    answer = "Yes"

                chip = ctx.locator(".chatbot_MessageContainer .chipsContainer .chatbot_Chip")
                text_input = ctx.locator(".chatbot_MessageContainer .textArea")
                radios = ctx.locator('.chatbot_MessageContainer input[type="radio"]')
                checkboxes = ctx.locator('.chatbot_MessageContainer input[type="checkbox"]')

                try:
                    if chip.count() > 0 and chip.first.is_visible(timeout=500):
                        chip.first.click()
                    elif radios.count() > 0:
                        radios.first.click(force=True)
                    elif checkboxes.count() > 0:
                        checkboxes.first.click(force=True)
                    elif text_input.is_visible(timeout=500):
                        text_input.fill(answer)
                    else:
                        continue
                except Exception:
                    continue

                send = ctx.locator(".sendMsg")
                try:
                    expect(send).to_be_enabled(timeout=3000)
                    send.click(timeout=3000)
                except Exception:
                    return False
                self.page.wait_for_timeout(800)

            if not handled:
                break
        return True

    def _answer_chatbot_if_visible(self):
        for ctx in self._chatbot_contexts():
            try:
                if ctx.locator(".chatbot_MessageContainer").is_visible(timeout=500):
                    self._answer_chatbot_heuristic()
                    try:
                        self.cba.classify_new_question()
                    except Exception as exc:
                        log_info(f"[WARN] ML chatbot fallback skipped: {exc}")
                    return True
            except Exception:
                continue
        return False

    def init_browser(self):
        log_info("[INFO] Launching browser...")
        self.playwright = sync_playwright().start()
        args = ["--disable-blink-features=AutomationControlled"]
        self.browser = self.playwright.chromium.launch(headless=self.headless, args=args)
        context_kwargs = {
            "user_agent": CHROMIUM_USER_AGENT,
            "viewport": {"width": 1366, "height": 768},
            "locale": "en-IN",
        }
        if self.has_valid_storage_state():
            context_kwargs["storage_state"] = self.storage_state_path
            log_info(f"[INFO] Loaded saved session from {self.storage_state_path}")
        else:
            log_info("[WARN] No saved session file; password login may require OTP in CI.")
        self.context = self.browser.new_context(**context_kwargs)
        self.context.grant_permissions([], origin="https://www.naukri.com")
        self.page = self.context.new_page()
        self.page.set_default_timeout(30000)
        self.cba = ChatbotAgent(self.page, self.username)

    def has_valid_storage_state(self):
        if not self.storage_state_path:
            return False

        path = Path(self.storage_state_path)
        if not path.exists() or not path.is_file():
            return False

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("storage state root must be an object")
            if "cookies" not in data or "origins" not in data:
                raise ValueError("storage state missing cookies/origins")
            if not isinstance(data["cookies"], list) or not isinstance(data["origins"], list):
                raise ValueError("storage state cookies/origins must be lists")
            return True
        except Exception as exc:
            log_info(f"[WARN] Ignoring invalid storage state at {path}: {exc}")
            return False

    def _session_is_authenticated(self):
        try:
            current_url = self.page.url
            # Logged-in Naukri user should be on a member page and show account/profile UI
            if current_url.startswith("https://www.naukri.com/mnjuser/homepage"):
                return True
            if current_url.startswith("https://www.naukri.com/mnjuser"):
                return True

            authenticated_selectors = (
                'text=/My Naukri/i',
                'text=/Logout/i',
                'text=/Sign out/i',
                'text=/Hello/i',
                'text=/Profile/i',
            )
            for selector in authenticated_selectors:
                try:
                    if self.page.locator(selector).first.is_visible(timeout=1000):
                        return True
                except Exception:
                    continue

            # If login fields are visible, the session is not authenticated.
            login_selectors = (
                '#usernameField',
                'input[placeholder="Enter Email ID / Username"]',
                'input[placeholder="Enter your active Email ID / Username"]',
                'input[placeholder="Enter Password"]',
                'input[placeholder="Enter your password"]',
                'text=/Login/i',
                'text=/Sign in/i',
            )
            for selector in login_selectors:
                try:
                    if self.page.locator(selector).first.is_visible(timeout=1000):
                        return False
                except Exception:
                    continue

            return False
        except Exception:
            return False

    def _restore_existing_session(self):
        if not self.has_valid_storage_state():
            return False

        try:
            self.page.goto("https://www.naukri.com/mnjuser/homepage", timeout=40000)
            self.page.wait_for_load_state('domcontentloaded')
            body_text = self.page.locator('body').inner_text()[:500]
            if 'Access Denied' in body_text:
                log_info("[WARN] Saved session is not usable in this environment; falling back to login.")
                return False
            if self._session_is_authenticated():
                self.save_storage_state()
                log_info("[INFO] Reused saved browser session; skipping manual login.")
                return True
            log_info("[WARN] Saved session cookies did not authenticate; falling back to login.")
            return False
        except Exception:
            return False

    def _ensure_authenticated_session(self):
        if self._session_is_authenticated():
            return True

        log_info("[WARN] Session authentication check failed; retrying with homepage restore.")
        if self._restore_existing_session():
            return True

        return False

    def save_storage_state(self):
        if not self.context or not self.save_storage_state_path:
            return
        output_path = Path(self.save_storage_state_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.context.storage_state(path=str(output_path))

    def dismiss_cookie_banner(self):
        selectors = [
            'text=Got it',
            'button:has-text("Got it")',
            'button[aria-label="Close"]',
        ]
        for selector in selectors:
            try:
                locator = self.page.locator(selector)
                if locator.first.is_visible(timeout=2000):
                    locator.first.click()
                    self.page.wait_for_timeout(500)
                    return True
            except Exception:
                continue
        return False

    def _otp_required(self):
        selectors = [
            'input[autocomplete="one-time-code"]',
            'input[placeholder*="OTP" i]',
            'input[type="tel"]',
            'text=/one.?time password|enter otp|verify otp/i',
        ]
        for selector in selectors:
            try:
                locator = self.page.locator(selector).first
                if locator.is_visible(timeout=1500):
                    return True
            except Exception:
                continue
        try:
            body_text = self.page.locator("body").inner_text(timeout=2000).lower()
        except Exception:
            return False
        return any(token in body_text for token in ("otp", "one-time password", "verify mobile"))

    def login(self):
        try:
            if self._restore_existing_session():
                return True

            log_info("[INFO] Starting password login...")
            self.page.goto("https://login.naukri.com/nLogin/Login.php", timeout=40000)
            self.page.wait_for_load_state('domcontentloaded')
            body_text = self.page.locator('body').inner_text()[:500]
            if 'Access Denied' in body_text:
                print("[ERROR] Access Denied: Naukri is blocking this browser or environment. No login form available.")
                self.page.screenshot(path="login_access_denied.png")
                print("[DEBUG] Screenshot saved as login_access_denied.png.")
                return False

            try:
                self.page.wait_for_selector(
                    '#usernameField, input[placeholder="Enter Email ID / Username"], input[placeholder="Enter your active Email ID / Username"]',
                    timeout=15000,
                )
            except Exception:
                print("[ERROR] Login form not found: The login page did not load the expected username field. This may be due to bot detection, network issues, or browser incompatibility.")
                self.page.screenshot(path="login_form_missing.png")
                print("[DEBUG] Screenshot saved as login_form_missing.png.")
                return False

            self.dismiss_cookie_banner()

            if self._click_google_login_button():
                if self._handle_google_login_prompt():
                    if self._session_is_authenticated():
                        self.save_storage_state()
                        log_info("[INFO] Google login successful! Starting job applications...")
                        return True
                    log_info("[WARN] Google sign-in flow completed but authentication was not confirmed.")

            username = self.page.locator(
                '#usernameField, input[placeholder="Enter Email ID / Username"], input[placeholder="Enter your active Email ID / Username"]'
            ).first
            password = self.page.locator(
                '#passwordField, input[placeholder="Enter Password"], input[placeholder="Enter your password"]'
            ).first
            submit = self.page.locator('button.blue-btn, button[type="submit"]').first

            username.fill(self.usr[0])
            password.fill(self.usr[1])
            submit.click()
            self.page.wait_for_timeout(5000)

            current_url = self.page.url
            if current_url.startswith("https://www.naukri.com/mnjuser/homepage"):
                self.save_storage_state()
                log_info("[INFO] Login successful! Starting job applications...")
                return True

            invalid_details = self.page.locator('text=Invalid details. Please check the Email ID - Password combination.').is_visible(timeout=3000)
            if invalid_details:
                log_info("Login failed: invalid email/password combination.")
                self.page.screenshot(path="login_failed.png")
                log_info("[DEBUG] Screenshot saved as login_failed.png.")
                return False

            if self._otp_required():
                if not self.otp:
                    log_info(
                        "[ERROR] Login requires OTP. For GitHub Actions, set the "
                        "JOBAUTO_STORAGE_STATE secret with a fresh session export "
                        "(see scripts/export_storage_state.py). Scheduled runs cannot wait for OTP."
                    )
                    self.page.screenshot(path="login_failed.png")
                    log_info("[DEBUG] Screenshot saved as login_failed.png.")
                    return False
                otp_input = self.page.locator(
                    'input[autocomplete="one-time-code"], input[placeholder*="OTP" i], input[type="tel"]'
                ).first
                otp_input.fill(self.otp)
                self.page.locator('button[type="submit"], button.blue-btn').first.click()
                self.page.wait_for_timeout(5000)
                if self.page.url.startswith("https://www.naukri.com/mnjuser/homepage"):
                    self.save_storage_state()
                    log_info("[INFO] Login successful! Starting job applications...")
                    return True

            log_info(f"Login failed. Current URL: {current_url}")
            self.page.screenshot(path="login_failed.png")
            print("[DEBUG] Screenshot saved as login_failed.png.")
            return False
        except Exception as e:
            log_info(f"Error during login: {e}")
            self.page.screenshot(path="login_exception.png")
            log_info("[DEBUG] Screenshot saved as login_exception.png.")
            return False

    def checkbox_apply(self):
        try:
            checkboxes = []
            for selector in (
                '.naukicon-ot-checkbox',
                '.srp-jobtuple-wrapper input[type="checkbox"]',
                '.jobTuple input[type="checkbox"]',
                '.list-jobtuple input[type="checkbox"]',
            ):
                found = self.page.locator(selector).element_handles()
                if found:
                    checkboxes = found
                    break
            log_info(f"[INFO] Found {len(checkboxes)} bulk-apply checkboxes on page.")
            if not len(checkboxes) == 0:
                lcbxs = 0
                for checkbox in checkboxes[:5]:
                    checkbox.scroll_into_view_if_needed()
                    checkbox.click(force=True)
                    lcbxs += 1
                apply_button = self.page.locator('.multi-apply-button')
                try:
                    with self.page.expect_response(
                        lambda response: "apply-workflow" in response.url,
                        timeout=10000,
                    ) as response_info:
                        apply_button.click(force=True)
                    response = response_info.value
                    if response.status == 403:
                        print("[INFO] Daily quota exceeded. Stopping application run.")
                        return {"status": "quota_exceeded", "clicked": lcbxs}
                except Exception:
                    pass
                try:
                    expect(self.page.locator(".chatbot_MessageContainer")).to_be_visible(timeout=3000)
                except:
                    try:
                        expect(self.page).to_have_url(self.pattern)
                        return {"status": "done", "clicked": lcbxs}
                    except:
                        raise
                return {"status": "underway", "found": len(checkboxes), "clicked": lcbxs}
            else:
                return {"status": "finished", "found": len(checkboxes), "clicked": 0}
        except Exception:
            return {"status": "failed"}

    def _job_already_applied(self):
        contexts = self._search_contexts()
        contexts = self._search_contexts()
        markers = (
            "text=/already applied/i",
            "text=/you have applied/i",
            "text=/application submitted/i",
            "text=/your application has been submitted/i",
            "text=/application submitted successfully/i",
            "text=/applied successfully/i",
            "text=/thank you for applying/i",
            "text=/application received/i",
            "text=/applied to this job/i",
            "button:has-text('Applied')",
        )
        for ctx in contexts:
            for selector in markers:
                try:
                    if ctx.locator(selector).first.is_visible(timeout=1500):
                        return True
                except Exception:
                    continue
        return False

    def _click_google_login_button(self):
        pages = self._all_pages()
        for ctx in pages:
            for selector in (
                'button:has-text("Sign in with Google")',
                'button:has-text("Sign in with google")',
                'button:has-text("Login with Google")',
                'button:has-text("Continue with Google")',
                'text=/sign in with google/i',
                'text=/continue with google/i',
            ):
                try:
                    btn = ctx.locator(selector).first
                    if btn.is_visible(timeout=1500):
                        log_info("[INFO] Google sign-in button detected; clicking it.")
                        btn.click(force=True)
                        ctx.wait_for_timeout(3000)
                        return True
                except Exception:
                    continue
        return False

    def _handle_google_login_prompt(self):
        pages = self._all_pages()
        found_prompt = False
        for ctx in pages:
            try:
                prompt = ctx.locator(
                    'button:has-text("Continue with Google"), '
                    'button:has-text("Sign in with Google"), '
                    'text=/continue with google/i, '
                    'text=/sign in with google/i'
                ).first
                if prompt.is_visible(timeout=1500):
                    found_prompt = True
                    log_info("[INFO] Google sign-in prompt detected; attempting account selection.")
                    prompt.click(force=True)
                    ctx.wait_for_timeout(3000)
            except Exception:
                continue

        account_selectors = (
            f'button:has-text("{self.username}")',
            f'div[role="button"]:has-text("{self.username}")',
            f'span:has-text("{self.username}")',
            'text=/@gmail\.com/i',
        )
        for ctx in pages:
            for selector in account_selectors:
                try:
                    account = ctx.locator(selector).first
                    if account.is_visible(timeout=3000):
                        account.click(force=True)
                        ctx.wait_for_timeout(4000)
                        log_info("[INFO] Selected Google account for sign-in.")
                        return True
                except Exception:
                    continue

        if found_prompt:
            log_info("[WARN] Google sign-in prompt was detected but account selection was not possible.")
        return False

    def _find_apply_button(self):
        """Locate the main job Apply control on a job description page."""
        primary = (
            "#apply-button",
            '[data-test-id="applyBtn"]',
            "button.apply-button-label",
            ".styles_JDC__apply-button button",
        )
        for selector in primary:
            try:
                btn = self.page.locator(selector).first
                if not btn.is_visible(timeout=2500):
                    continue
                text = " ".join(btn.inner_text().split()).lower()
                if "company site" in text or text in ("applied", "already applied"):
                    continue
                if re.search(r"\bapply\b", text):
                    return btn
            except Exception:
                continue

        extra_selectors = (
            'button:has-text("Apply")',
            'a:has-text("Apply")',
            'button:has-text("Apply Now")',
            'a:has-text("Apply Now")',
            'button[data-test-id="applyBtn"]',
            'button[data-automation="applyBtn"]',
        )
        for selector in extra_selectors:
            try:
                btn = self.page.locator(selector).first
                if not btn.is_visible(timeout=2500):
                    continue
                text = " ".join(btn.inner_text().split()).lower()
                if "company site" in text or text in ("applied", "already applied"):
                    continue
                return btn
            except Exception:
                continue

        try:
            for btn in self.page.get_by_role("button", name=re.compile(r"apply", re.I)).all():
                try:
                    if not btn.is_visible(timeout=500):
                        continue
                    text = " ".join(btn.inner_text().split()).lower()
                    if "company site" in text:
                        continue
                    if text in ("applied", "already applied"):
                        continue
                    if re.search(r"\bapply\b", text):
                        return btn
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _complete_apply_after_click(self, timeout_ms=40000):
        """Wait for chatbot or success after clicking Apply.

        CI can be slower; use a larger default timeout to reduce false negatives
        where the apply succeeds but success UI/URL appears slightly later.
        """

        deadline = time.time() + (timeout_ms / 1000)
        while time.time() < deadline:
            pages = self._all_pages()
            if self._job_already_applied():
                return True
            try:
                if self._any_page_matches_success_url():
                    return True
            except Exception:
                pass

            if self._handle_google_login_prompt():
                continue

            success_selectors = (
                "text=/your application has been submitted/i",
                "text=/application submitted successfully/i",
                "text=/application received/i",
                "text=/applied successfully/i",
                "text=/thank you for applying/i",
                "button:has-text('Applied')",
                "text=/Application submitted/i",
                "text=/Applied/i",
            )

            for ctx in pages:
                for selector in success_selectors:
                    try:
                        if ctx.locator(selector).is_visible(timeout=500):
                            return True
                    except Exception:
                        continue

            chatbot = self.page.locator(".chatbot_MessageContainer")
            try:
                if chatbot.is_visible(timeout=500):
                    self._answer_chatbot_heuristic()
                    try:
                        self.cba.classify_new_question()
                    except Exception as exc:
                        log_info(f"[WARN] Chatbot model step: {exc}")
                    if self._job_already_applied():
                        return True
                    if not chatbot.is_visible(timeout=1000):
                        return True
            except Exception:
                pass

            chatbot = self.page.locator(".chatbot_MessageContainer")
            try:
                if chatbot.is_visible(timeout=500):
                    self._answer_chatbot_heuristic()
                    try:
                        self.cba.classify_new_question()
                    except Exception as exc:
                        log_info(f"[WARN] Chatbot model step: {exc}")
                    if self._job_already_applied():
                        return True
                    if not chatbot.is_visible(timeout=1000):
                        return True
            except Exception:
                pass

            self.page.wait_for_timeout(500)

        confirmed = self._job_already_applied()
        if not confirmed:
            try:
                body_text = self.page.locator("body").inner_text(timeout=2000)
            except Exception:
                body_text = ""
            clean_body = body_text[:500].replace("\n", " ")
            log_info(f"[WARN] Apply click not confirmed. URLs={self._all_page_urls()}")
            log_info(f"[WARN] Body excerpt after apply: {clean_body}")
            try:
                screenshot_path = f"apply_failed_{int(time.time())}.png"
                self.page.screenshot(path=screenshot_path)
                log_info(f"[DEBUG] Saved failed apply screenshot to {screenshot_path}")
            except Exception:
                pass
        return confirmed

    def apply_(self):
        """Per-job apply loop for --apply --filters (same as original working flow)."""
        chatbot_timeout = 20000
        try:
            self.page.wait_for_load_state('load', timeout=10000)
        except Exception:
            pass
        self.page.wait_for_timeout(2000)
        job_links = self.page.eval_on_selector_all(
            '.title',
            'elements => elements.map(element => element.getAttribute("href")) .filter(href => href !==null)'
        )
        log_info(
            f"[INFO] Page {self.page_no}: found {len(job_links)} job links "
            f"(applied {self.applied_count}/{self.applno})"
        )
        for job_index, jl in enumerate(job_links, start=1):
            if self.applied_count >= self.applno:
                log_info(f"\n✅ Successfully applied to {self.applied_count} jobs!")
                break
            if job_index == 1 or job_index % 5 == 0:
                log_info(f"[INFO] Processing job {job_index}/{len(job_links)} on page {self.page_no}...")
            try:
                self.page.wait_for_timeout(1000)
                self.page.goto(jl, timeout=30000)
                try:
                    self.page.wait_for_load_state('load', timeout=10000)
                except Exception:
                    pass
                self.page.wait_for_timeout(2500)

                if self._job_already_applied():
                    log_info(f"[SKIP] Job {job_index}: already applied")
                    continue

                if self._handle_google_login_prompt():
                    self.page.wait_for_timeout(3000)
                    self.page.wait_for_load_state('load', timeout=10000)

                apply_button = self._find_apply_button()
                if not apply_button:
                    try:
                        self.page.evaluate("window.scrollTo(0, 0)")
                        self.page.wait_for_timeout(300)
                        self.page.evaluate("window.scrollBy(0, 600)")
                        self.page.wait_for_timeout(500)
                    except Exception:
                        pass
                    apply_button = self._find_apply_button()

                if not apply_button:
                    try:
                        if self.page.locator('button:has-text("Apply on company site")').first.is_visible(timeout=1000):
                            log_info(f"[SKIP] Job {job_index}: company-site apply only")
                            continue
                    except Exception:
                        pass
                    log_info(f"[SKIP] Job {job_index}: no Apply button found")
                    continue

                try:
                    button_text = " ".join(apply_button.inner_text().split())
                    log_info(f"[DEBUG] Clicking apply button: {button_text}")
                except Exception:
                    pass
                try:
                    apply_button.scroll_into_view_if_needed(timeout=3000)
                except Exception:
                    pass
                apply_button.click(force=True)
                try:
                    self.page.wait_for_load_state('networkidle', timeout=5000)
                except Exception:
                    pass
                self.page.wait_for_timeout(1000)

                if self._complete_apply_after_click(timeout_ms=chatbot_timeout):
                    self.applied_count += 1
                    log_info(f"✅ Applied to {self.applied_count}/{self.applno} jobs.")
                else:
                    log_info(f"[SKIP] Job {job_index}: Apply not confirmed")
            except Exception as e:
                log_info(f"[SKIP] Job {job_index}: error ({e})")
                continue

        if self.applied_count < self.applno:
            if self.page_no >= 15:
                log_info("[WARN] Reached page 15 without hitting target apply count.")
                return
            self.page_no += 1
            parsed = urlparse(self.base_page_url)
            new_path = parsed.path + f"-{self.page_no}"
            modified_url = urlunparse(parsed._replace(path=new_path))
            try:
                self.page.goto(modified_url)
            except Exception:
                log_info(f"\n✅ Successfully applied to {self.applied_count} jobs!")
                return
            self.apply_()

    def filter_apply(self, s, e='', l='', ja='3'):
        self.search = s
        if not self.search:
            log_info("Search keyword required")
            return {"response": "search required", "applied": 0}
        self.experience = e
        self.experience_years = str(e) if e not in (None, "") else "2"
        self.location = l
        self.jobage = ja
        log_info(f"[INFO] Starting apply run (target {self.applno} jobs)...")
        self.init_browser()
        try:
            if not self.login():
                log_info("Login failed, aborting job application.")
                return {"response": "login failed", "applied": 0}
            time.sleep(1)
            log_info("[INFO] Applying search filters...")
            self.filter_()
            try:
                self.page.wait_for_load_state('networkidle', timeout=20000)
            except Exception:
                pass
            self.base_page_url = self.page.url
            self.page_no = 1
            if not self._ensure_authenticated_session():
                log_info(f"[ERROR] Saved session is not authenticated after applying filters; aborting apply run. Current URL: {self.page.url}")
                try:
                    body_excerpt = self.page.locator('body').inner_text()[:500]
                    log_info(f"[DEBUG] Page body excerpt: {body_excerpt}")
                except Exception:
                    pass
                return {"response": "session invalid", "applied": 0}
            log_info(f"[INFO] Search results ready: {self.base_page_url}")
            self.apply_()
        except Exception as e:
            log_info(f"[ERROR] Error during filter_apply: {e}")
        finally:
            self.close()
        log_info(f"[FINAL] Job application completed. Applied to {self.applied_count} jobs total.")
        return {"response": "applied successfully", "applied": self.applied_count}

    def filter_(self):
        serch = self.page.locator(".nI-gNb-sb__icon-wrapper")
        serch.click() 
        self.page.locator('input[placeholder="Enter keyword / designation / companies"]').type(self.search,delay=100)
        if self.location:
            self.page.locator('input[placeholder="Enter location"]').type(self.location,delay=100)
        if self.experience:
            self.page.locator('#experienceDD').click()
            self.page.locator(f'li[index="{self.experience}"]').click()
        serch.click()
        self.page.wait_for_load_state('load')
        curl = self.page.url 
        if self.jobage:
            nurl = curl+f"&jobAge={self.jobage}"
            self.page.goto(nurl)

    def start_apply(self,tab):
        self.tabIndex = 0
        self.tab = tab
        self.init_browser()
        try:
            if self.login():
                botactions = self.bot_actions()
                return botactions
        finally:
            self.close()
        
    def bot_actions(self):
        try:
            time.sleep(2)
            self.page.click('.nI-gNb-menuItems__anchorDropdown')
            if not self.tab=="profile":
                self.page.click(f"#{self.tab}")
            self.page.wait_for_load_state("networkidle")
            if self.applied_count >= self.applno:
                print(f"applied {self.applied_count} jobs")
                return {"response":"applied successfully","applied":self.applied_count}
            else:
                cbapl = self.checkbox_apply()
            if cbapl["status"] == 'failed':
                print(f"finished daily quota with {self.applied_count} jobs")
                self.close()
                return {"response":"quota finished","applied":self.applied_count}
            elif cbapl["status"] == 'quota_exceeded':
                print("[INFO] Naukri reported that the daily quota has been exceeded.")
                self.close()
                return {"response":"quota exceeded","applied":self.applied_count}
            elif cbapl["status"] == 'done':
                self.applied_count += cbapl["clicked"]
                self.bot_actions()
            elif cbapl["status"] == 'underway':
                self.cba.classify_new_question()
                try:
                    expect(self.page).to_have_url(self.pattern)
                    self.applied_count += cbapl["clicked"]
                    self.bot_actions()
                except Exception as e:
                    print("An error occured answering naukri questions :===>",e)
                    self.close()
                    return {"response":"error on botactions","error":str(e)}
            elif cbapl['status'] == "finished":
                self.tabIndex += 1
                self.tab = self.tabs[self.tabIndex]
                self.bot_actions()
        except Exception as e:
            self.close()
            print(f"applied {self.applied_count} jobs but an error occured :===>{str(e)}")

    def close(self):
        try:
            self.save_storage_state()
        except Exception:
            pass
        try:
            if self.page is not None:
                self.page.close()
        except Exception:
            pass
        try:
            if self.context is not None:
                self.context.close()
        except Exception:
            pass
        try:
            if self.browser is not None:
                self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright is not None:
                self.playwright.stop()
        except Exception:
            pass
