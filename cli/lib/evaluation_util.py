import os, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data","golden_dataset.json")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data","movies.json")

def load_golden_set() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)["test_cases"]

def load_movies():
    with open(MOVIES_PATH, "r") as file:
        return json.load(file)