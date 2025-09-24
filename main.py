from flask import Flask, request, jsonify
from joblib import load
import re
from nltk.stem import WordNetLemmatizer

# ---- LemmaTokenizer Definition ----
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    
    def __call__(self, reviews):
        tokens = re.findall(r'\b\w+\b', reviews.lower())
        return [self.wordnetlemma.lemmatize(word) for word in tokens]

# ---- Initialize Flask App ----
app = Flask(__name__)

# ---- Load Vectorizer and Model ----
# Make sure 'vectorizer.joblib' and 'model.joblib' are in the same folder as main.py
vectorizer = load('vectorizer.joblib')
model = load('model.joblib')

# ---- Define Routes ----
@app.route('/')
def home():
    return "Flask App is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        
        # Transform text using the vectorizer
        features = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return jsonify({'prediction': str(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---- Run Flask App ----
if __name__ == "__main__":
    # Use debug=True only for local testing
    app.run(debug=False, host='0.0.0.0', port=5000)
