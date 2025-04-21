from flask import Flask, render_template, request, redirect, url_for
import random
import pickle
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

app = Flask(__name__)

# Define the articles and categories
articles = [
    ("Latest Tech Innovations in AI", "technology"),
    ("Top 10 Healthy Habits", "health"),
    ("Champions League Highlights", "sports"),
    ("Advancements in Robotics", "technology"),
    ("Yoga for Mental Wellness", "health"),
    ("Top Scorers in NBA 2024", "sports")
]

categories = ["technology", "health", "sports"]
category_to_label = {cat: i for i, cat in enumerate(categories)}
label_to_category = {i: cat for cat, i in category_to_label.items()}

# Load or initialize the model and data
def load_model():
    if os.path.exists("model.pkl") and os.path.exists("X.npy") and os.path.exists("y.npy"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        X = np.load("X.npy")
        y = np.load("y.npy")
    else:
        model = LogisticRegression()
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y = np.array([0, 1, 2])
        model.fit(X, y)
        save_model(model, X, y)
    return model, X, y

def save_model(model, X, y):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save("X.npy", X)
    np.save("y.npy", y)

model, X, y = load_model()

@app.route("/")
def index():
    displayed_articles = random.sample(articles, 3)
    return render_template("index.html", articles=displayed_articles)

@app.route("/click", methods=["POST"])
def click():
    category = request.form.get("category")
    features = [0] * len(categories)
    features[category_to_label[category]] = 1
    global X, y, model
    X = np.vstack([X, features])
    y = np.append(y, category_to_label[category])
    model.fit(X, y)
    save_model(model, X, y)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
