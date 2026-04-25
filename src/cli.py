"""
TARA CLI
===========
Command-line interface for quick queries and pipeline testing.
"""

import argparse
import sys
import json

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        prog="tara",
        description="TARA - Telecom Agentic RAG Assistant CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- query command ---
    query_parser = subparsers.add_parser("query", help="Ask a telecom question")
    query_parser.add_argument("text", type=str, help="Your question")
    query_parser.add_argument("--top-k", type=int, default=10, help="Number of sources")
    query_parser.add_argument("--no-rerank", action="store_true", help="Skip reranking")

    # --- analyze command (route-only, no LLM needed) ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a query without LLM")
    analyze_parser.add_argument("text", type=str, help="Query to analyze")

    # --- index command ---
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--source", type=str, required=True, help="Path to PDF directory")

    # --- eval command ---
    eval_parser = subparsers.add_parser("eval", help="Run TeleQnA evaluation")
    eval_parser.add_argument("--dataset", type=str, default="data/raw/teleqna/TeleQnA.json")
    eval_parser.add_argument("--max-questions", type=int, default=None)

    # --- health command ---
    subparsers.add_parser("health", help="Check system health")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "analyze":
        _handle_analyze(args.text)
    elif args.command == "query":
        _handle_query(args.text, args.top_k, not args.no_rerank)
    elif args.command == "health":
        _handle_health()
    elif args.command == "index":
        _handle_index(args.source)
    elif args.command == "eval":
        _handle_eval(args.dataset, args.max_questions)
    else:
        parser.print_help()


def _handle_analyze(query: str):
    """Analyze a query — shows routing, intent, entities. No LLM needed."""
    from src.query.router import QueryRouter

    router = QueryRouter()
    routed = router.route(query)

    print("\n" + "=" * 60)
    print("  [TARA] Query Analysis")
    print("=" * 60)
    print(f"  Original:   {routed.original_query}")
    print(f"  Enriched:   {routed.enriched_query}")
    print(f"  Intent:     {routed.intent.value}")
    print(f"  Complexity: {routed.complexity:.2f}")
    print(f"  Path:       {routed.path.value}")
    print(f"\n  Entities:")
    ent = routed.entities
    if ent.spec_refs:
        print(f"    Spec refs:    {', '.join(ent.spec_refs)}")
    if ent.kpis:
        print(f"    KPIs:         {', '.join(ent.kpis)}")
    if ent.procedures:
        print(f"    Procedures:   {', '.join(ent.procedures)}")
    if ent.technologies:
        print(f"    Technologies: {', '.join(ent.technologies)}")
    if ent.cell_ids:
        print(f"    Cell IDs:     {', '.join(ent.cell_ids)}")
    print("=" * 60 + "\n")


def _handle_query(query: str, top_k: int, use_reranker: bool):
    """Full query pipeline — requires Ollama + Qdrant."""
    try:
        from src.retrieval.retrieval_pipeline import RetrievalPipeline
        from src.generation.generator import Generator
        from src.generation.post_processor import PostProcessor

        pipeline = RetrievalPipeline()
        docs = pipeline.retrieve(query, top_k=top_k, use_reranker=use_reranker)

        generator = Generator()
        response = generator.generate(query, docs)

        processor = PostProcessor()
        processed = processor.process(response.text, total_sources=len(docs))

        print("\n" + "=" * 60)
        print("  [TARA] Response")
        print("=" * 60)
        print(f"\n{processed.answer}\n")
        print(f"  Sources cited: {processed.source_count}")
        print(f"  Grounded:      {'YES' if processed.is_grounded else 'NO'}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n  [!] Error: {e}")
        print("  Make sure Ollama is running and Qdrant index exists.\n")


def _handle_health():
    """Check system health."""
    from src.generation.llm_interface import OllamaClient

    print("\n" + "=" * 60)
    print("  [TARA] Health Check")
    print("=" * 60)

    # Ollama
    client = OllamaClient()
    ollama_ok = client.is_available()
    print(f"  Ollama:  {'[OK] Connected' if ollama_ok else '[X] Unavailable'}")

    # Qdrant
    try:
        from pathlib import Path
        from config.settings import get_settings
        settings = get_settings()
        qdrant_exists = Path(settings.qdrant.path).exists()
        print(f"  Qdrant:  {'[OK] Index found' if qdrant_exists else '[X] No index'}")
    except Exception:
        print(f"  Qdrant:  [X] Error checking")

    print("=" * 60 + "\n")


def _handle_index(source_dir: str):
    """Index documents from a directory."""
    from src.ingestion.pdf_extractor import PDFExtractor
    from src.ingestion.chunker import TelecomChunker
    from src.ingestion.telecom_preprocessor import TelecomPreprocessor
    from src.ingestion.indexer import Indexer

    extractor = PDFExtractor()
    preprocessor = TelecomPreprocessor()
    chunker = TelecomChunker(chunk_size=512, chunk_overlap=50, respect_sections=True)
    indexer = Indexer()

    print(f"\n  Indexing documents from: {source_dir}")

    docs = extractor.extract_directory(source_dir)
    print(f"  Extracted {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        clean_text = preprocessor.preprocess(doc.full_text)
        chunks = chunker.chunk_document(clean_text, metadata=doc.metadata)
        all_chunks.extend(chunks)

    print(f"  Created {len(all_chunks)} chunks")

    count = indexer.index_chunks(all_chunks)
    print(f"  Indexed {count} chunks into Qdrant\n")


def _handle_eval(dataset_path: str, max_questions: int):
    """Run TeleQnA evaluation."""
    from src.evaluation.teleqna_evaluator import TeleQnAEvaluator

    evaluator = TeleQnAEvaluator(dataset_path)
    loaded = evaluator.load_dataset()

    if loaded == 0:
        print("\n  [!] No questions loaded. Download TeleQnA first.\n")
        return

    print(f"\n  Loaded {loaded} questions. Evaluation requires a running TARA pipeline.\n")


if __name__ == "__main__":
    main()
