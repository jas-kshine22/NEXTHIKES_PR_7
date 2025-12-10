from flask import Flask, render_template, request
import pickle
import re
import string

# Load trained model and TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Flask app
app = Flask(__name__)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)                # remove mentions/hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()             # remove extra spaces
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tweet = request.form["tweet"]

    # Clean the tweet
    cleaned = clean_text(tweet)

    # Convert to vector
    vector = tfidf.transform([cleaned])

    # Predict
    prediction = model.predict(vector)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
