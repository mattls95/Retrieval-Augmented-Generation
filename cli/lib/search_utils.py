import os
import json
import string
from nltk.stem import PorterStemmer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data","movies.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
    
def load_movies():
    with open(MOVIES_PATH, "r") as file:
        return json.load(file)
    

def tokenization(text: str) -> list:
    text = process_string(text)
    words = text.split()
    stopwords = load_stopwords()
    filtered_words = list(filter(lambda word: word not in stopwords, words))
    stemmed_words = []
    stemmer = PorterStemmer()
    for token in filtered_words:
        stemmed_words.append(stemmer.stem(token))
    return stemmed_words

def process_string(query) -> str:
    punctuations = {}
    for char in string.punctuation:
        punctuations[char] = None

    processed_query = query.translate(str.maketrans(punctuations)).lower()
    return processed_query