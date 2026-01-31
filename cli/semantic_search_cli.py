#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, cmd_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser(name="verify", help="Verify model")
    subparsers.add_parser(name="verify_embeddings", help="Verify embeddings")

    embed_text_parser = subparsers.add_parser(name="embed_text", help="Generate embedding for supplied text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    embed_query_parser = subparsers.add_parser(name="embedquery", help="Generate embedding for query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embedding for")

    search_parser = subparsers.add_parser(name="search", help="Search for similairities")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int,  default=5, help="Limit for query return") 

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            cmd_search(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()