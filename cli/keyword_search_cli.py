#!/usr/bin/env python3

import argparse
from lib.inverted_index import InvertedIndex
from lib import search
from lib.search_utils import BM25_K1
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

    idf_parser = subparsers.add_parser("idf", help="Get Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term to get the inverse document frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TFIDF Score")
    tfidf_parser.add_argument("doc_id", type=int, help="Doc ID to get the TFIDF for")
    tfidf_parser.add_argument("term", type=str, help="Term to get the TFIDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
  "bm25tf", help="Get BM25 TF score for a given document ID and term"
)
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

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
        case "idf":
            try:
                inverted_index.load()
                idf = inverted_index.get_idf(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except ValueError:
                    print("more terms than expected given expected: 1")
        case "tfidf":
            try:
                inverted_index.load()
                tf_idf = inverted_index.get_tf_idf(args.doc_id, args.term)
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except ValueError:
                    print("more terms than expected given expected: 1")
        case "bm25idf":
            try:
                bm25idf = inverted_index.bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except ValueError:
                    print("more terms than expected given expected: 1")
        case "bm25tf":
            try:
                bm25tf = inverted_index.bm25_tf_command(args.doc_id, args.term)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except ValueError:
                    print("more terms than expected given expected: 1")                      

if __name__ == "__main__":
    main()