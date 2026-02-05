import argparse
from lib.hybrid_search import cmd_normalize_score, HybridSearch
from lib.search_utils import load_movies, gemini_query_spell, gemini_query_rewrite, gemini_query_expand,gemini_query_rerank, gemini_query_batch
import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import time, json
from sentence_transformers import CrossEncoder

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
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking most relevant documents")

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
            initial_limit = args.limit
            if args.rerank_method:
                limit = initial_limit * 5
                results = hybrid_search.rrf_search(args.query, args.k, limit)
                if args.rerank_method == "individual":
                    for doc in results:
                        doc["rerank_score"] = generate_gemini_response(args.query, args.rerank_method, doc)
                        time.sleep(5)
                    results_rerank = sorted(results, key=lambda doc: doc["rerank_score"], reverse=True)[:initial_limit]
                    for i, doc in enumerate(results_rerank, start=1):
                        print(f"{i}. {doc['title']}\n"
                        f"Rerank Score: {doc['rerank_score']}\n"
                        f"RRF Score: {doc['rrf_score']}\n"
                        f"BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}\n"
                        f"{doc['document'][:200]}\n")
                    return
                elif args.rerank_method == "batch":
                    json_list =  generate_gemini_response(args.query, args.rerank_method, "doc", results)
                    print("RAW LLM RESPONSE:", repr(json_list))  # debug
                    ranks = json.loads(json_list)
                    
                    rank_lookup = {}
                    position = 0
                    for doc_id in ranks:
                        rank_lookup[doc_id] = position
                        position += 1
                    results_rerank_batch = sorted(results, key=lambda doc: rank_lookup[doc["id"]])[:initial_limit]
                    print("Reranking top 3 results using batch method...\n")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
                    for i, doc in enumerate(results_rerank_batch, start=1):
                        print(f"{i}. {doc['title']}\n"
                        f"Rerank rank: {i}\n"
                        f"RRF Score: {doc['rrf_score']}\n"
                        f"BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}\n"
                        f"{doc['document'][:200]}\n")
                elif args.rerank_method == "cross_encoder":
                    pairs = []
                    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                    for doc in results:
                        pairs.append([args.query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
                    scores = cross_encoder.predict(pairs)
                    for i, doc in enumerate(results):
                        doc["cross_encoder_score"] = scores[i]

                    results_cross = sorted(results, key=lambda doc: doc["cross_encoder_score"], reverse=True)[:initial_limit]
                    for i, doc in enumerate(results_cross, start=1):
                        print(f"{i}. {doc['title']}\n"
                        f"Cross Encoder Score: {doc['cross_encoder_score']}\n"
                        f"RRF Score: {doc['rrf_score']}\n"
                        f"BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}\n"
                        f"{doc['document'][:200]}\n")
                    


            if args.enhance:
                enhanced_query = generate_gemini_response(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                results = hybrid_search.rrf_search(enhanced_query, args.k, limit)
            else:
                results = hybrid_search.rrf_search(args.query, args.k, limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}\n"
                    f"RRF Score: {result['rrf_score']}\n"
                    f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n"
                    f"{result['document'][:200]}\n")
        case _:
            parser.print_help()

def generate_gemini_response(query, choice, doc, doc_list_str):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    try:
        if choice == "spell":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_spell(query))
        elif choice == "rewrite":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_rewrite(query))
        elif choice == "expand":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_expand(query))
        elif choice == "individual":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_rerank(query, doc))
        elif choice == "batch":
            content = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query_batch(query, doc_list_str))
    except genai_errors.ClientError:
        return query
    return content.text

if __name__ == "__main__":
    main()