import os
import json
import random
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer


with open("./jab/data/user_data.json","r") as file:
    user_data = json.load(file)
def training(data):
    def skills(skill,exp):
        op = {
                "patterns":[
                    f"How many years of experience do you have in {skill} ?",
                    f"Experience in {skill}",
                    f"How long have you been working with {skill} ?"
                    ],
                "tag":skill+"_exp",
                "answer":f"{exp}"
            }
        return op
    training_data = [
        {
            "patterns": ["What is your full name?", "Tell me your full name.", "What is your name?", "Can I have your full name?"],
            "tag": "full_name",
            "answer": data['Full name']
        },
        {   "patterns":["what is your gender?","what do you identify as?","Gender"],
            "tag":"gender",
            "answer":data['Gender']
        },
        {
            "patterns": ["What is your mobile number?", "Tell me your phone number.", "What is cell phone number?", "Can I have your mobile phone number?"],
            "tag": "mobile",
            "answer": data['Mobile']
        },
        {
            "patterns": ["What is your email address?", "Tell me your email.", "What is your email?", "Can I have your email?"],
            "tag": "email",
            "answer": data['Email']
        },
        {
            "patterns": ["Where are you located?", "What is your location?", "Where do you live?", "Your current city?","which state are you from?"],
            "tag": "location",
            "answer": data['Location']
        },
        {
            "patterns": ["What is your current salary?", "What is your CTC?", "How much are you earning?", "What is your income?"],
            "tag": "current_salary",
            "answer": data['Salary']
        },
        {
            "patterns": ["What is your first name?", "Tell me your first name.", "What is your given name?", "Can I know your first name?"],
            "tag": "first_name",
            "answer": data['First name']
        },
        {
            "patterns": ["What is your last name?", "What is your surname?", "Tell me your last name.", "Your family name?"],
            "tag": "last_name",
            "answer": data['Last name']
        },
        {
            "patterns": ["What is your date of birth?", "When is your birthday?", "What is your DOB?", "Your birth date?"],
            "tag": "date_of_birth",
            "answer": data['Date of birth']
        },
        {
            "patterns": ["What is your expected salary?", "What is your desired CTC?", "What salary are you expecting?", "Your expected income?"],
            "tag": "expected_salary",
            "answer": data['Expected salary']
        },
        {
            "patterns": ["What is your preferred work location?", "Where do you prefer to work?", "What is your desired job location?", "Your preferred city for work?"],
            "tag": "preferred_location",
            "answer": data['Preferred work location']
        },
        {
            "patterns": ["What is your notice period?", "When can you join?", "What is your availability to join?", "How soon can you start?"],
            "tag": "notice_period",
            "answer": data['Notice period']
        },
        {
            "patterns": ["How many years of experience do you have?","What is your current experience?", "How much experience do you have?", "Your total work experience?", "Current professional experience?"],
            "tag": "current_experience",
            "answer": data['Total experience']
        },
        {
            "patterns": [f"Have you worked at our organization?", f"Were you previously employed by organization?", f"Did you work for organization?"],
            "tag": "bool_employee",
            "answer": 'No'
        },
        {
            "patterns": ["How should we address you?", "What is your salutation?", "What is your title?", "How do you prefer to be addressed?"],
            "tag": "salutation",
            "answer": data['Salutation']
        },
        {
        "patterns": ["I consent to the Privacy Policy", "Do you agree to the Privacy Policy?", "I accept the Privacy Policy", "Consent to the Privacy Policy"],
        "tag": "privacy_policy_consent",
        "answer": 'Yes'
        },
        {
        "patterns": [
            "Are you currently residing in location or willing to relocate to location?", 
            "Do you live in location or are you open to relocating?", 
            "Are you based in location or would you move to location?", 
            "Are you willing to move to location?",
            "Do you live in location or are you considering relocating?"
        ],
        "tag": "relocation_bool",
        "answer": "yes"
        },
     ]
    if data["Skills"]!="":
        training_data.extend([skills(s,e) for s,e in data['Skills'].items()])
        training_data.extend([{
            "patterns": ["What are your key skills?", "Tell me your skills.", "What skills do you have?", "Your key competencies?"],
            "tag": "Skills",
            "answer": ', '.join([str(i) for i in data['Skills'].keys()])
        }])
    return training_data

class ChatbotBuild:
    def __init__(self, username):
        self.username = username
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_words = set(['?', '!', '.', ','])
        self.user_data = user_data
        self.patterns = []
        self.tags = set()
        self.responses = {}
        self.words = []
        self.classes = []
        self.documents = []
        self.model = None
        self.training_data = training(self.user_data)
        self.load_data()
        self.preprocess_data()
        self.build_model()

    def load_data(self):
        for intent in self.training_data:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.patterns.append((word_list, intent['tag']))
                self.tags.add(intent['tag'])
            self.responses[intent['tag']] = intent['answer']
    def preprocess_data(self):
        words_temp = []
        classes_set = set()
        documents_lemma = []
        
        # Collect and lemmatize words once
        for pattern in self.patterns:
            for word in pattern[0]:
                if word not in self.ignore_words:
                    lemmatized = self.lemmatizer.lemmatize(word.lower())
                    words_temp.append(lemmatized)
            classes_set.add(pattern[1])
            
            # Store lemmatized pattern
            lemmatized_pattern = [self.lemmatizer.lemmatize(word.lower()) for word in pattern[0] if word not in self.ignore_words]
            documents_lemma.append((lemmatized_pattern, pattern[1]))
        
        self.words = sorted(set(words_temp))
        self.classes = sorted(classes_set)
        words_set = set(self.words)
        
        # Create training data using set operations
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in documents_lemma:
            word_set = set(doc[0])
            bag = [1 if word in word_set else 0 for word in self.words]
            output_row = output_empty.copy()
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        
        random.shuffle(training)
        training = np.array(training, dtype=object)
        self.train_x = np.array(list(training[:, 0]), dtype=np.float32)
        self.train_y = np.array(list(training[:, 1]), dtype=np.float32)

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(len(self.train_x[0]),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        adam = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.train_x, self.train_y, epochs=100, batch_size=32, verbose=1)
        if not os.path.exists(f"./jab/data/{self.username}"):
            os.mkdir(f"./jab/data/{self.username}")
        self.model.save(f'./jab/data/{self.username}/model.keras')
        print("model built and saved!")
