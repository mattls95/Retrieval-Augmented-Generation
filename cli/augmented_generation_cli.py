import argparse
from lib.evaluation_util import load_movies
from lib.hybrid_search import HybridSearch
from google import genai
from dotenv import load_dotenv
import os

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--k", type=int, default=60, help="Weight for ranking")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results")
    summarize_parser.add_argument("query", type=str, help="Query for summary")
    summarize_parser.add_argument("--limit", default=5, help="Limit for search results")
    summarize_parser.add_argument("--k", type=int, default=60, help="Weight for ranking")

    citations_parser = subparsers.add_parser("citations", help="Generate citations for search results")
    citations_parser.add_argument("query", type=str, help="Query for summary")
    citations_parser.add_argument("--limit", default=5, help="Limit for search results")
    citations_parser.add_argument("--k", type=int, default=60, help="Weight for ranking")

    question_parser = subparsers.add_parser("question", help="Generate citations for search results")
    question_parser.add_argument("query", type=str, help="Query for summary")
    question_parser.add_argument("--limit", default=5, help="Limit for search results")
    question_parser.add_argument("--k", type=int, default=60, help="Weight for ranking")

    args = parser.parse_args()
    movies = load_movies()
    hybrid_search = HybridSearch(movies["movies"])
    match args.command:
        case "rag":
            query = args.query
            results = hybrid_search.rrf_search(query, args.k, 5)
            response = gemini_1(query, results)

            print("Search Results:")
            for result in results:
                print(f"\t- {result["title"]}")
            print()
            print(response)
        case "summarize":
            results = hybrid_search.rrf_search(query=args.query, k=args.k, limit=args.limit)
            response = gemini_2(args.query, results)

            print("Search Results:")
            for result in results:
                print(f"\t- {result["title"]}")
            print()
            print(response)
        case "citations":
            results = hybrid_search.rrf_search(query=args.query, k=args.k, limit=args.limit)
            response = gemini_3(args.query, results)

            print("Search Results:")
            for result in results:
                print(f"\t- {result["title"]}")
            print()
            print(response)
        case "question":
            results = hybrid_search.rrf_search(query=args.query, k=args.k, limit=args.limit)
            response = gemini_4(args.query, results)

            print("Search Results:")
            for result in results:
                print(f"\t- {result["title"]}")
            print()
            print(response)
        case _:
            parser.print_help()

def gemini_4(question, context):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    content = client.models.generate_content(model="gemini-2.5-flash", contents=get_prompt_4(question, context))
    return content.text

def get_prompt_4(question, context):
    return f"""Answer the user's question based on the provided movies that are available on Hoopla.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Question: {question}

            Documents:
            {context}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
def gemini_3(query, documents):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    content = client.models.generate_content(model="gemini-2.5-flash", contents=get_prompt_3(query, documents))
    return content.text

def get_prompt_3(query, documents):
    return f"""Answer the question or provide information based on the provided documents.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                Query: {query}

                Documents:
                {documents}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources using [1], [2], etc. format when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""
def gemini_2(query, results):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    content = client.models.generate_content(model="gemini-2.5-flash", contents=get_prompt_2(query, results))
    return content.text

def get_prompt_2(query, results):
    return f"""
            Provide information useful to this query by synthesizing information from multiple search results in detail.
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Search Results:
            {results}
            Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
            """

def gemini_1(query, docs):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    content = client.models.generate_content(model="gemini-2.5-flash", contents=get_prompt_1(query, docs))
    return content.text

def get_prompt_1(query, docs):
    return f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {docs}

    Provide a comprehensive answer that addresses the query:"""

if __name__ == "__main__":
    main()
