
from .search_utils import load_movies, CACHE_DIR,INDEX_PATH,DOCMAP_PATH, tokenization, TERM_FREQUENCIES_PATH
from pickle import dump, load
from collections import Counter
import os

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        tokenized_text = tokenization(text)
        counter = Counter()
        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)
            counter[token] += 1
        self.term_frequencies[doc_id] = counter

    def get_document(self, term):
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)

    def get_tf(self, doc_id, term) -> int:
        token = tokenization(term)   
        if len(token) > 1:
            raise ValueError("more then one token given")
        
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term)

    def build(self):
        movies = load_movies()
        for movie in movies["movies"]:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            dump(self.index, f)

        with open(DOCMAP_PATH, "wb") as f:
            dump(self.docmap, f)

        with open(TERM_FREQUENCIES_PATH, "wb") as f:
            dump(self.term_frequencies, f)

    def load(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCMAP_PATH) or not os.path.exists(TERM_FREQUENCIES_PATH):
            raise FileNotFoundError("index or docmap or term frequencies file not found")
        else:
            with open(INDEX_PATH,"rb") as f:
                self.index = load(f)
            with open(DOCMAP_PATH, "rb") as f:
                self.docmap = load(f)
            with open(TERM_FREQUENCIES_PATH, "rb") as f:
                self.term_frequencies = load(f)

    