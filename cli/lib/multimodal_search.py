from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from lib import search_utils

class MultiModalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path=image_path)

        results = []
        for i, embedding in enumerate(self.text_embeddings):
            similarity_score = cosine_similarity(embedding, image_embedding)
            results.append({
                        "doc_id": self.documents[i]["id"],
                        "title": self.documents[i]["title"],
                        "description": self.documents[i]["description"],
                        "similarity_score" : similarity_score
                            })
            
        return sorted(results, key=lambda item: item["similarity_score"], reverse=True)[:5]


    def embed_image(self, image_path):
        return self.model.encode([Image.open(image_path)])[0]
    

def image_search_cmd(image_path):
    movies = search_utils.load_movies()["movies"]
    multi_modal_search = MultiModalSearch(movies)
    return multi_modal_search.search_with_image(image_path)
    
def verify_image_embedding(image_path):
    multi_modal_search = MultiModalSearch()
    embedding = multi_modal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)