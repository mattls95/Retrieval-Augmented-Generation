#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, cmd_search, cmd_chunck,cmd_sematic_chunk 
from lib.chunked_semantic_search import cmd_embed_chunks, cmd_search_chunked

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

    chunk_parser = subparsers.add_parser(name="chunk", help="Split query into chunks")
    chunk_parser.add_argument("text", type=str, help="Text to split into chunks")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Default chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Default option for overlapping words")

    semantic_chunk_parser = subparsers.add_parser(name="semantic_chunk", help="Splits based on sentence boundaries")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to split into chunks")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max chunk size to split one")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Default option for overlapping words")

    subparsers.add_parser(name="embed_chunks", help="Generate chunk embeddings")

    search_chunked_parser = subparsers.add_parser(name="search_chunked", help="Search for chunked query")   
    search_chunked_parser.add_argument("query", type=str, help="Query for search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Limit for results defaults to 5")

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
        case "chunk":
            cmd_chunck(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            results = cmd_sematic_chunk(args.text, args.max_chunk_size, args.overlap)
            for i, text in enumerate(results, start=1):
                print(f"{i}. {text}")
        case "embed_chunks":
            cmd_embed_chunks()
        case "search_chunked":
            cmd_search_chunked(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
