
from .search_utils import load_movies, CACHE_DIR,INDEX_PATH,DOCMAP_PATH, tokenization
from pickle import dump, load
import os

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokenized_text = tokenization(text)
        for token in set(tokenized_text):
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)

    def get_document(self, term):
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)
                                

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

    def load(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCMAP_PATH):
            raise FileNotFoundError("index or docmap file not found")
        else:
            with open(INDEX_PATH,"rb") as f:
                self.index = load(f)
            with open(DOCMAP_PATH, "rb") as f:
                self.docmap = load(f)