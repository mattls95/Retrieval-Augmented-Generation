import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data","movies.json")
CHUNK_EMBEDDINGS = os.path.join(PROJECT_ROOT, "cache/chunk_embeddings.npy")
CHUNK_METADATA = os.path.join(PROJECT_ROOT, "cache/chunk_metadata.json")

def embeddings_file_exist():
    return os.path.exists(MOVIE_EMBEDDINGS_PATH)

def file_exist(path):
    return os.path.exists(path)

def load_movies():
    with open(MOVIES_PATH, "r") as file:
        return json.load(file)