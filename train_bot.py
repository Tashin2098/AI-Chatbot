import json
import random
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare data
X = []
y = []
responses = {}

for intent in data["intents"]:
    tag = intent["tag"]
    responses[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(tag)

# Preprocessing
def preprocess(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

X = [preprocess(sentence) for sentence in X]

# Vectorize
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save model and components
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(responses, open("responses.pkl", "wb"))

print("✅ Model training complete. Files saved!")
