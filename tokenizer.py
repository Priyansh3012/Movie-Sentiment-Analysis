import re
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    
    def __call__(self, reviews):
        tokens = re.findall(r'\b\w+\b', reviews.lower())
        return [self.wordnetlemma.lemmatize(word) for word in tokens]
