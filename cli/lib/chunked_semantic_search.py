from lib.semantic_search import SemanticSearch, cmd_sematic_chunk, cosine_similarity
from lib.semantic_search_util import CHUNK_METADATA, CHUNK_EMBEDDINGS, file_exist, load_movies
import numpy as np 
import json as json


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        all_chunks = []
        chunk_metadata = []

        for i, doc in enumerate(documents):
            movie_desc = doc['description']
            if len(movie_desc) == 0:
                continue
            chunks = cmd_sematic_chunk(movie_desc, 4, 1)
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(chunks)
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(CHUNK_EMBEDDINGS, self.chunk_embeddings)
        with open(CHUNK_METADATA, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if file_exist(CHUNK_EMBEDDINGS) and file_exist(CHUNK_METADATA):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS)
            with open(CHUNK_METADATA, "r") as f:
                meta = json.load(f)
                self.chunk_metadata = meta["chunks"]

            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings) :
            cosine_score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append({
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": cosine_score
            })
        idx_to_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if movie_idx not in idx_to_scores:
                idx_to_scores[movie_idx] = chunk_score
            elif chunk_score["score"] > idx_to_scores[movie_idx]["score"]:
               idx_to_scores[movie_idx] = chunk_score

        best_chunk_scores = list(idx_to_scores.values())
        best_chunk_scores.sort(key=lambda cs: cs["score"], reverse=True)
        top_chunk_scores = best_chunk_scores[:limit]

        results =[]
        for chunk_score in top_chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            doc = self.documents[movie_idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": chunk_score["score"],
                "metadata": doc.get("metadata") or {}
            })
        return results


def cmd_search_chunked(query, limit,):
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(movies['movies'])
    results = chunked_semantic_search.search_chunks(query, limit)

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
        print(f"   {result["document"]}...")

def cmd_embed_chunks():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies['movies'])
    print(f"Generated {len(embeddings)} chunked embeddings")


