import pickle
import random
import string

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
responses = pickle.load(open("responses.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_response(user_input):
    user_input = preprocess(user_input)
    X_test = vectorizer.transform([user_input])
    tag = model.predict(X_test)[0]
    return random.choice(responses[tag])

print("🤖 ChatBot is running! (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("ChatBot: Goodbye!")
        break
    reply = get_response(user_input)
    print("ChatBot:", reply)
