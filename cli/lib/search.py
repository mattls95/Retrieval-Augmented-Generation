import json
import os
import string

def search(title) -> None:
    file_path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/movies.json"))
    with open(file_path_abs, "r") as file:
        movies = json.load(file)

    results = []

    processed_title = process_string(title)

    for movie in movies["movies"]:
        if processed_title in process_string(movie["title"]):
            results.append(movie)

    results = sorted(results,key=lambda movie: movie["id"])
    results = results[:5]
    for movie in results:
        print(f"{movie["id"]}. {movie["title"]} {movie["id"]}")

def process_string(query) -> str:
    punctiations = {}
    for char in string.punctuation:
        punctiations[char] = None

    processed_query = query.translate(str.maketrans(punctiations)).lower()
    return processed_query