import time
import re
import numpy as np
import nltk
import json
from urllib.parse import urlparse, urlunparse
from playwright.sync_api import sync_playwright , expect
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

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
    def __init__(self, usreml, usrpas, username, number=10, headless=False):
        self.browser = None
        self.page = None
        self.usr = [usreml, usrpas]
        self.username = username
        self.applno = number
        self.applied_count = 0
        self.page_no = 1
        self.tabs = ["profile", "apply", "preference", "similar_jobs"]
        self.pattern = re.compile(r'https://.*/myapply/saveApply\?strJobsarr=')
        self.headless = headless

    def init_browser(self):
        playwright = sync_playwright().start()
        args = ["--disable-blink-features=AutomationControlled"]
        self.browser = playwright.chromium.launch(headless=self.headless, args=args)
        context = self.browser.new_context()
        context.grant_permissions([], origin="https://www.naukri.com")
        self.page = context.new_page()
        self.page.set_default_timeout(30000)
        self.cba = ChatbotAgent(self.page, self.username)

    def login(self):
        try:
            self.page.goto("https://www.naukri.com", timeout=40000)
            self.page.wait_for_load_state('networkidle')
            self.page.click('a[title="Jobseeker Login"]:visible, #login_Layer:visible')
            self.page.wait_for_timeout(1000)
            self.page.fill('input[type="text"], input[name="username"]', self.usr[0])
            self.page.fill('input[type="password"]', self.usr[1])
            self.page.click('button[type="submit"], button:has-text("Login")')
            self.page.wait_for_timeout(2000)  # Give time for redirect
            current_url = self.page.url
            if current_url.startswith("https://www.naukri.com/mnjuser/homepage"):
                print("[INFO] Login successful! Starting job applications...")
                return True
            else:
                print(f"Login failed. Current URL: {current_url}")
                self.page.screenshot(path="login_failed.png")
                print(f"[DEBUG] Screenshot saved as login_failed.png.")
                return False
        except Exception as e:
            print(f"Error during login: {e}")
            self.page.screenshot(path="login_exception.png")
            print(f"[DEBUG] Screenshot saved as login_exception.png.")
            return False
        
    def checkbox_apply(self):
        try:
            checkboxes = self.page.locator('.naukicon-ot-checkbox').element_handles()
            print(f"Found {len(checkboxes)} checkboxes.")
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

    def apply_(self):
        try:
            self.page.wait_for_load_state('load', timeout=10000)
        except Exception as e:
            pass
        self.page.wait_for_timeout(2000)  # Wait for JS to render jobs
        job_links = self.page.eval_on_selector_all(
            '.title',
            'elements => elements.map(element => element.getAttribute("href")) .filter(href => href !==null)'
        )
        print(f"[INFO] Found {len(job_links)} job links, applying now...")
        for job_index, jl in enumerate(job_links, start=1):
            if self.applied_count >= self.applno:
                print(f"\n✅ Successfully applied to {self.applied_count} jobs!")
                break
            try:
                self.page.wait_for_timeout(1000)
                self.page.goto(jl, timeout=30000)
                try:
                    self.page.wait_for_load_state('load', timeout=10000)
                except:
                    pass
                self.page.wait_for_timeout(1500)
                
                # Check if this job has "Apply on company site" button - skip these
                company_site_buttons = self.page.locator('button:has-text("Apply on company site")').all()
                if company_site_buttons and len(company_site_buttons) > 0:
                    continue
                
                # Try multiple selectors specifically for the "Apply" button (not "Apply on company site")
                apply_button = None
                selectors = [
                    'button:has-text("Apply"):not(:has-text("company"))',
                    'button:has-text("Apply")',
                    '#apply-button',
                    '.apply-button',
                    '[data-test-id="applyBtn"]',
                    'button.apply-button-label',
                ]
                
                for selector in selectors:
                    try:
                        if apply_button is None:
                            try:
                                elems = self.page.locator(selector).all()
                                for elem in elems:
                                    try:
                                        button_text = elem.inner_text()
                                        # Check if it's exactly "Apply" and not "Apply on company site"
                                        if button_text.strip() == "Apply" and "company site" not in button_text.lower():
                                            if elem.is_visible(timeout=2000):
                                                apply_button = elem
                                                break
                                    except:
                                        pass
                                if apply_button:
                                    break
                            except:
                                pass
                    except:
                        pass
                
                if not apply_button:
                    # Try scrolling and looking again
                    try:
                        self.page.evaluate('window.scrollBy(0, 500)')
                        self.page.wait_for_timeout(500)
                        for selector in selectors:
                            try:
                                elem = self.page.locator(selector).first
                                if elem.is_visible(timeout=2000):
                                    apply_button = elem
                                    break
                            except:
                                pass
                    except:
                        pass
                
                if not apply_button:
                    continue
                
                apply_button.click()
                self.page.wait_for_timeout(1000)
                
                try:
                    expect(self.page.locator(".chatbot_MessageContainer")).to_be_visible(timeout=3000)
                    self.cba.classify_new_question()
                    self.applied_count+=1
                    print(f"✅ Applied to {self.applied_count} jobs.")
                except:
                    try:
                        expect(self.page).to_have_url(self.pattern)
                        self.applied_count+=1
                        print(f"✅ Applied to {self.applied_count} jobs.")
                    except:
                        continue
            except Exception as e:
                continue 
        if self.applied_count<self.applno:
            self.page_no+=1
            parsed = urlparse(self.base_page_url)
            new_path = parsed.path + f"-{self.page_no}"
            modified_url = urlunparse(parsed._replace(path=new_path))
            try:
                self.page.goto(modified_url)
            except:
                print(f"\n✅ Successfully applied to {self.applied_count} jobs!")
                return
            self.apply_()

    def filter_apply(self, s, e='', l='', ja='3'):
        self.search = s
        if not self.search:
            print("Search keyword required")
            return
        self.experience = e
        self.location = l
        self.jobage = ja
        self.init_browser()
        try:
            if not self.login():
                print("Login failed, aborting job application.")
                self.page.close()
                return {"response": "login failed", "applied": 0}
            time.sleep(1)
            self.filter_()
            self.base_page_url = self.page.url
            self.apply_()
        except Exception as e:
            print(f"[ERROR] Error during filter_apply: {e}")
        finally:
            try:
                self.page.close()
            except:
                pass
        print(f"[FINAL] Job application completed. Applied to {self.applied_count} jobs total.")
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
        if self.login():
            botactions = self.bot_actions()
            return botactions
        
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
            if self.page is not None:
                self.page.close()
        except Exception:
            pass
        try:
            if self.browser is not None:
                self.browser.close()
        except Exception:
            pass
