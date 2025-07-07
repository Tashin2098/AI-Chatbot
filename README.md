# 🤖 AI Chatbot using Flask & Scikit-learn

This is a simple rule-based AI chatbot built using Python, Flask, and Scikit-learn. It uses intent classification with logistic regression and bag-of-words vectorization.

---

## 🚀 Features

- Intent classification using `LogisticRegression`
- Text preprocessing with `CountVectorizer`
- Responses managed via `intents.json`
- REST API built with Flask
- Frontend chat interface using HTML/CSS/JS
- Easily extendable with new intents and patterns

---

## 📁 Project Structure

ai_chatbot/
│
├── app.py # Flask server
├── chatbot.py # Command-line version
├── train_bot.py # Script to train model
├── intents.json # Intent patterns and responses
├── model.pkl # Trained classification model
├── vectorizer.pkl # Fitted vectorizer
├── responses.pkl # Saved intent responses
├── templates/
│ └── index.html # Frontend UI
└── static/
└── style.css # Optional styling
