from flask import Flask, render_template, request
import os
import nltk
from joblib import load
from tokenizer import LemmaTokenizer  # import custom class

# Ensure LemmaTokenizer is visible for unpickling
globals()['LemmaTokenizer'] = LemmaTokenizer

# NLTK path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Load models
vectorizer = load('vectorizer.joblib')
model = load('modelLogReg.joblib')

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        transformed_review = vectorizer.transform([review]).toarray()
        prediction = model.predict(transformed_review)
        sentiment = 'Positive' if prediction == '1' else 'Negative'
        return render_template('index.html', review=review, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
