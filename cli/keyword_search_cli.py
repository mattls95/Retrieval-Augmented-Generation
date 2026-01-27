#!/usr/bin/env python3

import argparse
from lib.inverted_index import InvertedIndex
from lib import search
import os

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get the frequency for")

    args = parser.parse_args()
    inverted_index = InvertedIndex()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                inverted_index.load()
            except FileNotFoundError:
                    print("Index not found, please run the build command first.")
                    os._exit(1)
            search.search(args.query, inverted_index)
        case "build":
            inverted_index.build()
            inverted_index.save()
        case "tf":
            try:
                inverted_index.load()
                print(inverted_index.get_tf(args.doc_id, args.term))
            except ValueError:
                    print("more terms than expected given expected: 1")

if __name__ == "__main__":
    main()