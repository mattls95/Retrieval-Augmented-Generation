import argparse
from lib.evaluation_util import load_golden_set, load_movies
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    hybrid_search = HybridSearch(load_movies()["movies"])
    args = parser.parse_args()
    limit = args.limit

    test_cases = load_golden_set()
    for test_case in test_cases:
        query = test_case["query"]
        retrieved_docs = hybrid_search.rrf_search(query=query, k=60, limit=limit)
        
        retrieved_titles =[]
        for doc in retrieved_docs:
            retrieved_titles.append(doc['title'])

        relevant_titles = test_case["relevant_docs"]

        relevant_retrieved_count = 0
        for title in retrieved_titles:
            if title in relevant_titles:
                relevant_retrieved_count += 1

        precision = relevant_retrieved_count / len(retrieved_titles)
        recall = relevant_retrieved_count / len(relevant_titles)
        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(relevant_titles)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}")
        print()

if __name__ == "__main__":
    main()