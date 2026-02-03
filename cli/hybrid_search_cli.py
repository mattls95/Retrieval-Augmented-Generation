import argparse
from lib.hybrid_search import cmd_normalize_score, HybridSearch
from lib.search_utils import load_movies

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
            results = hybrid_search.rrf_search(args.query, args.k, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}\n"
                      f"RRF Score: {result['rrf_score']}\n"
                      f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n"
                      f"{result['document'][:200]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()