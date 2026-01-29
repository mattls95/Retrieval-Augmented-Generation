
from .search_utils import load_movies, CACHE_DIR,INDEX_PATH,DOCMAP_PATH, tokenization, TERM_FREQUENCIES_PATH, BM25_K1, DOC_LENGTH_PATH, BM25_B, DOC_LENGTH_PATH
from pickle import dump, load
from collections import Counter
import os, math

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokenized_text = tokenization(text)
        counter = Counter()
        self.doc_lengths[doc_id] = len(tokenized_text)
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
    
    def bm25(self, doc_id, term):
        bm_tf = self.__get_bm25_tf(doc_id, term)
        bm_idf = self.__get_bm25_idf(term)
        return bm_tf * bm_idf
    
    def bm25_search(self, query, limit):
        tokens = tokenization(query)
        scores = {}
    
        for doc_id in self.docmap:
            for token in tokens:
                if doc_id not in scores:
                    scores[doc_id] = self.bm25(doc_id, token)
                else:
                    scores[doc_id] += self.bm25(doc_id, token)

        scores_sorted = sorted(scores.items(), key= lambda score: score[1], reverse=True)
        results = []
        for doc_id, score in scores_sorted[:limit]:
            doc = self.docmap[doc_id]
            results.append((doc, score))
        return results


    def get_tf(self, doc_id, term) -> int:
        tokens = tokenization(term)   
        if len(tokens) > 1:
            raise ValueError("more then one token given")
        
        token = tokens[0]

        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(token, 0)
    
    def get_idf(self, term):
        tokens = tokenization(term)
        if len(tokens) > 1:
            raise ValueError("more then one token given")
        
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
    def get_tf_idf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term)
    
    def __get_bm25_idf(self, term: str) -> float:
        tokens = tokenization(term)
        if len(tokens) > 1:
            raise ValueError("more then one token given")
        
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
    
    def bm25_idf_command(self, term):
        self.load()
        return self.__get_bm25_idf(term)
    
    def bm25_tf_command(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        self.load()
        return self.__get_bm25_tf(doc_id, term, k1, b)
    
    def __get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm = 1 -b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        raw_tf = self.get_tf(doc_id,term)
        bm25_saturation = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return bm25_saturation
    
    def __get_avg_doc_length(self) -> float:
        total_length = 0
        for doc_id in self.doc_lengths:
            total_length += self.doc_lengths[doc_id]
        if len(self.doc_lengths) == 0:
            return 0.0
        return total_length / len(self.doc_lengths)

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

        with open(DOC_LENGTH_PATH, "wb") as f:
            dump(self.doc_lengths, f)

    def load(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCMAP_PATH) or not os.path.exists(TERM_FREQUENCIES_PATH) or not os.path.exists(DOC_LENGTH_PATH):
            raise FileNotFoundError("index or docmap or term frequencies file not found")
        else:
            with open(INDEX_PATH,"rb") as f:
                self.index = load(f)
            with open(DOCMAP_PATH, "rb") as f:
                self.docmap = load(f)
            with open(TERM_FREQUENCIES_PATH, "rb") as f:
                self.term_frequencies = load(f)
            with open(DOC_LENGTH_PATH, "rb") as f:
                self.doc_lengths = load(f)

    