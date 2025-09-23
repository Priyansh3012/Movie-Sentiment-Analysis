from flask import Flask, render_template, request
import nltk
from joblib import load
from tokenizer import LemmaTokenizer  # your tokenizer.py
import os
import shutil
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ----------------------------
# Monkey-patch punkt_tab issue
# ----------------------------
try:
    find('tokenizers/punkt_tab/english.pickle')
except LookupError:
    nltk_data_dir = 'nltk_data'  # folder in your project root
    os.makedirs(os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab'), exist_ok=True)
    try:
        punkt_src = find('tokenizers/punkt/english.pickle')
        punkt_dst = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english.pickle')
        shutil.copy(punkt_src, punkt_dst)
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)

# Download wordnet if not present
try:
    find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='nltk_data')

# ----------------------------
# Load models
# ----------------------------
vectorizer = load('vectorizer.joblib')
model = load('modelLogReg.joblib')

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']

        # Transform the review using the loaded vectorizer
        transformed_review = vectorizer.transform([review]).toarray()

        # Predict sentiment
        prediction = model.predict(transformed_review)
        sentiment = 'Positive' if prediction == '1' else 'Negative'

        return render_template('index.html', review=review, sentiment=sentiment)

    return render_template('index.html')


if __name__ == '__main__':
    nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
