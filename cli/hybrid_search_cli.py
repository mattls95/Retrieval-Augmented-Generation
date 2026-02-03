import argparse
from lib.hybrid_search import cmd_normalize_score, HybridSearch
from lib.search_utils import load_movies, gemini_query_spell, gemini_query_rewrite, gemini_query_expand
import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser =  parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="scores to normalize")

    weighted_parser = subparser.add_parser("weighted-search", help="Perform weighted search")
    weighted_parser.add_argument("query", type=str, help="Query to search for")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter")
    weighted_parser.add_argument("--limit", type=int, default=5)

    rrf_search_parser = subparser.add_parser(name="rrf-search", help="Query for search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search for")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="Weight for ranking")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit for search results default to 5")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")

    args = parser.parse_args()
    hybrid_search = HybridSearch(load_movies()["movies"])
    match args.command:
        case "normalize":
            cmd_normalize_score(args.scores)
        case "weighted-search":
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i} {result["title"]}\nHybrid Score: {result["hybrid_score"]}\n BM25: {result["bm25_score"]}, Semantic:{result["semantic_score"]}\n{result["document"]}")
        case "rrf-search":
            print("test")
            if args.enhance:
                enhanced_query = generate_gemini_response(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                results = hybrid_search.rrf_search(enhanced_query, args.k, args.limit,)
            else:
                results = hybrid_search.rrf_search(args.query, args.k, args.limit)
                for i, result in enumerate(results, start=1):
                    print(f"{i}. {result['title']}\n"
                        f"RRF Score: {result['rrf_score']}\n"
                        f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n"
                        f"{result['document'][:200]}\n")
        case _:
            parser.print_help()

def generate_gemini_response(query, choice):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    print("Before call")
    try:
        if choice == "spell":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_spell(query))
        elif choice == "rewrite":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_rewrite(query))
        elif choice == "expand":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_expand(query))
    except genai_errors.ClientError:
        return query
    return content.text

if __name__ == "__main__":
    main()