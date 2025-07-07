from flask import Flask, render_template, request, jsonify
import pickle
import random
import string

app = Flask(__name__)

# Load model and data
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_response = get_response(user_message)
    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
