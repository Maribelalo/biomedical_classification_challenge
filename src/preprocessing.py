import re
import unicodedata
import nltk
from nltk.stem import SnowballStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STEMMER = SnowballStemmer('english')

def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(text):
    tokens = nltk.word_tokenize(clean_text(text))
    return ' '.join([STEMMER.stem(word) for word in tokens if word not in STOPWORDS and len(word) > 2])
