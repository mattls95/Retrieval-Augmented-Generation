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

    args = parser.parse_args()
    match args.command:
        case "normalize":
            cmd_normalize_score(args.scores)
        case "weighted-search":
            hybrid_search = HybridSearch(load_movies()["movies"])
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i} {result["title"]}\nHybrid Score: {result["hybrid_score"]}\n BM25: {result["bm25_score"]}, Semantic:{result["semantic_score"]}\n{result["document"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()