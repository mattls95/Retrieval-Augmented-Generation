import argparse
from lib import multimodal_search

def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_parser.add_argument("image_path", type=str, help="image path to verify")

    image_search_parser = subparsers.add_parser("image_search", help="Image search parser")
    image_search_parser.add_argument("image_path", type=str, help="Image search parser")

    args = parser.parse_args()

    #multimodal_search.verify_image_embedding(args.image_path)
    results = multimodal_search.image_search_cmd(args.image_path)
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result["title"]} (similarity: {result["similarity_score"]:.3f})\n"
              f"\t{result["description"][:200]}\n")

if __name__ == "__main__":
    main()