import os
import json
import string
from nltk.stem import PorterStemmer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data","movies.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTH_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")
BM25_K1 = 1.5
BM25_B = 0.75

def gemini_query_spell(query):
   return f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    Return only the corrected query, with no labels or quotes
    If no errors, return the original query.
    """

def gemini_query_rewrite(query):
   return f"""Rewrite this movie search query to be more specific and searchable.

            Original: "{query}"

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep it concise (under 10 words)
            - It should be a google style search query that's very specific
            - Don't use boolean logic

            Examples:

            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            Rewritten query:"""

def gemini_query_expand(query):
   return f"""Expand this movie search query with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            This will be appended to the original query.

            Examples:

            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            Query: "{query}"
            """

def gemini_query_rerank(query, doc):
    return f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""
def gemini_query_batch(query, doc_list_str):
    return f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return ONLY the IDs in order of relevance (best match first).
                Return ONLY a valid JSON list, nothing else. Do not add explanations.

                [75, 12, 34, 2, 1]
                """

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
    
def load_movies():
    with open(MOVIES_PATH, "r") as file:
        return json.load(file)
    
def tokenization(text: str) -> list:
    text = process_string(text)
    words = text.split()
    stopwords = load_stopwords()
    filtered_words = list(filter(lambda word: word not in stopwords, words))
    stemmed_words = []
    stemmer = PorterStemmer()
    for token in filtered_words:
        stemmed_words.append(stemmer.stem(token))
    return stemmed_words

def process_string(query) -> str:
    punctuations = {}
    for char in string.punctuation:
        punctuations[char] = None

    processed_query = query.translate(str.maketrans(punctuations)).lower()
    return processed_query