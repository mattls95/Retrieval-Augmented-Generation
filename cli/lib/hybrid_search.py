import os

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_search_result = self._bm25_search(query,limit * 500)
        chunked_search_result = self.semantic_search.search_chunks(query, limit * 500)

        bm25_normalized = self.__normalize_search_results(bm25_search_result)
        chunked_normalized = self.__normalize_search_results(chunked_search_result)

        combined_scores = {}

        for doc in bm25_normalized:
            doc_id = doc["id"]
            if doc_id not in combined_scores:
                combined_scores[doc["id"]] = {
                    "title": doc["title"],
                    "document": doc["document"],
                    "bm25_score": doc["normalized_score"]
                }
            combined_scores[doc_id]["bm25_score"] = doc["normalized_score"]

        for doc in chunked_normalized:
            doc_id = doc["id"]
            if doc_id not in combined_scores:
                combined_scores[doc["id"]] = {
                    "title": doc["title"],
                    "document": doc["document"],
                    "semantic_score": doc["normalized_score"]
                }
            combined_scores[doc_id]["semantic_score"] = doc["normalized_score"]
            
        for doc_id, data in combined_scores.items():
            hybrid_score_value = hybrid_score(data["bm25_score"], data["semantic_score"])
            combined_scores[doc_id]["hybrid_score"] = hybrid_score_value
            
        combined_scores = sorted(combined_scores.values(),key=lambda value : value["hybrid_score"], reverse=True)
        return combined_scores[:limit]

    def __normalize_search_results(self, results):
        scores = []
        for result in results:
            scores.append(result["score"])

        normalized = _normalize_score(scores)
        for i, result in enumerate(results):
            result["normalized_score"] = normalized[i]

        return results
    


    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def cmd_normalize_score(scores: list):
    _normalize_score(scores)

def _normalize_score(scores: list):
    results = []
    min_score = min(scores)
    max_score = max(scores)
    if not scores:
        return
    elif min_score == max_score:
        return [1.0] * len(scores)
    else:
        for score in scores:
            normalized_score = ((score - min_score) / (max_score - min_score))
            results.append(normalized_score)
    return results

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score