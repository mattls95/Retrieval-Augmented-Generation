from sentence_transformers import SentenceTransformer
import numpy as np
from lib import semantic_search_util as util
import re

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarity_score_document = []
        for i,_ in enumerate(self.documents):
            document = self.documents[i]
            embedding = self.embeddings[i]
            cos_sim = cosine_similarity(query_embedding, embedding)
            similarity_score_document.append((cos_sim, document))

        sorted_doc_score = sorted(similarity_score_document,key=lambda item: item[0], reverse=True)
        results = []
        for i in range(limit):
            results.append({
                "score": sorted_doc_score[i][0],
                "title": sorted_doc_score[i][1]["title"], 
                "description": sorted_doc_score[i][1]["description"]
                }
            )
        return results
    
    def build_embedding(self, documents):
        self.documents = documents
        document_lst = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            document_lst.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(document_lst, show_progress_bar=True)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        document_lst = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            document_lst.append(f"{doc['title']}: {doc['description']}")
        if util.embeddings_file_exist():
            self.embeddings = np.load(util.MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embedding(documents)

    def generate_embedding(self, text):
        if self.__is_text_valid(text):
            encode_param_lst = [text]
            return self.model.encode(encode_param_lst)[0]
        else:
            raise ValueError("Empty text")

    def __is_text_valid(self, text):
        if len(text) == 0 or text.isspace():
            return False
        return True
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def cmd_sematic_chunk(text, max_chunk_size, overlap):
    results = []
    stripped_text = text.strip()
    if len(stripped_text) == 0:
        return []
    sentences = re.split(r"(?<=[.!?])\s+",stripped_text)
    sentence = sentences[0]
    if len(sentences) == 1 and not (sentence.endswith(".") or sentence.endswith("!") or sentence.endswith("?")):
        results.append(sentence)
        return results   

    start = 0
    while start < len(sentences) - overlap:
        end = start + max_chunk_size
        chunck_sentences = sentences[start:end]
        chunk_sentence = " ".join(chunck_sentences).strip()
        if len(chunk_sentence) != 0:
            results.append(chunk_sentence)
        start += max_chunk_size - overlap
    return results

def cmd_chunck(text, chunk_size, overlap):
    num_char = len(text)
    words = text.split()
    results = []
    start = 0
    last_end = 0
    while start < len(words):
        end = start + chunk_size
        if min(end, len(words)) > last_end:
            last_end = end
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            results.append(chunk_text)
        start += (chunk_size - overlap)

    print(f"Chunking {num_char} characters")
    for i, text in enumerate(results, start=1):
        print(f"{i}. {text}")

def cmd_search(query, limit):
    movie_list = util.load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movie_list["movies"])
    results = semantic_search.search(query, limit)
    print(results)
    for i, element in enumerate(results, start=1):
        print(f"{i}. {element["title"]} (score: {element["score"]})\n\n{element["description"]}\n")
    
def verify_embeddings():
    semantic_search = SemanticSearch()
    movie_list = util.load_movies()
    semantic_search.load_or_create_embeddings(movie_list["movies"])
    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")

def embed_text(text):
    semantic_model = SemanticSearch()
    embedding = semantic_model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    semantic_model = SemanticSearch()
    embedding = semantic_model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_model():
    semantic_model = SemanticSearch()
    print(f"Model loaded: {semantic_model.model}")
    print(f"Max sequence length: {semantic_model.model.max_seq_length}")
