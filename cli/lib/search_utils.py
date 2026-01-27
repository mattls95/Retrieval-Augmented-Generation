import os
import json
from nltk.stem import PorterStemmer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data","movies.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
    
def load_movies():
    with open(MOVIES_PATH, "r") as file:
        return json.load(file)
    

def tokenization(query: str) -> list:
    words = query.split()
    stopwords = load_stopwords()
    filtered_words = list(filter(lambda word: word not in stopwords, words))
    stemmed_words = []
    stemmer = PorterStemmer()
    for token in filtered_words:
        stemmed_words.append(stemmer.stem(token))
    return stemmed_words