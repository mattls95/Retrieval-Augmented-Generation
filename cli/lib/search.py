import json
import os

def search(title) -> None:
    file_path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/movies.json"))
    with open(file_path_abs, "r") as file:
        movies = json.load(file)

    results = []

    for movie in movies["movies"]:
        if title.lower() in movie["title"].lower():
            results.append(movie)

    results = sorted(results,key=lambda movie: movie["id"])
    results = results[:5]
    for movie in results:
        print(f"{movie["id"]}. {movie["title"]} {movie["id"]}")