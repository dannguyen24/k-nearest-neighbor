from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import re

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))



def load_data(filepath):
    labels, reviews = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.replace('#EOF', '').strip()
            if not line:
                continue
            parts = line.split('\t', 1)  
            labels.append(int(parts[0]))
            reviews.append(parts[1])
    return labels, reviews

labels, reviews = load_data('train_data.txt')


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)        # remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]   # stop word removal
    tokens = [stemmer.stem(t) for t in tokens]             # stemming
    return ' '.join(tokens)

clean_reviews = [preprocess(r) for r in reviews]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_reviews)

svd = TruncatedSVD(n_components=100)            # try 50, 100, 200
X_reduced = svd.fit_transform(X)