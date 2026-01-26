import json
import os
import string

def search(query) -> None:
    file_path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/movies.json"))
    with open(file_path_abs, "r") as file:
        movies = json.load(file)

    results = []

    processed_query = process_string(query)
    query_tokens = tokenization(processed_query)

    for movie in movies["movies"]:
        title_tokens = tokenization(process_string(movie["title"]))
        for token in query_tokens:
            for word in title_tokens:
                if token in word:
                    results.append(movie)
                    break
            if movie in results:
                break

    results = sorted(results,key=lambda movie: movie["id"])
    results = results[:5]
    print_movies(results)

def process_string(query) -> str:
    punctiations = {}
    for char in string.punctuation:
        punctiations[char] = None

    processed_query = query.translate(str.maketrans(punctiations)).lower()
    return processed_query

def tokenization(query: str) -> list:
    words = query.split()
    return words

def print_movies(movies: list) -> None:
    for movie in movies:
        print(f"{movie["id"]}. {movie["title"]} {movie["id"]}")