import string

from .search_utils import load_movies, tokenization
from lib.inverted_index import InvertedIndex

def search(query, inverted_index:InvertedIndex) -> None:
    results = []
    seen_ids = set()
    query_tokens = tokenization(query)
    
    for token in query_tokens:
        doc_ids = inverted_index.get_document(token)
        for doc_id in doc_ids:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            movie = inverted_index.docmap[doc_id]
            if len(results) >= 5:
                break
            results.append(movie)
        if len(results) >= 5:
            break

    print_movies(results)

def process_string(query) -> str:
    punctiations = {}
    for char in string.punctuation:
        punctiations[char] = None

    processed_query = query.translate(str.maketrans(punctiations)).lower()
    return processed_query

def print_movies(movies: list) -> None:
    for movie in movies:
        print(f"{movie['id']}. {movie['title']} {movie['id']}")