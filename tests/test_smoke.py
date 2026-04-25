"""
Smoke tests — verify modules import and basic classes work without dependencies.
"""

import pytest


def test_settings_import():
    """Config settings should import without error."""
    from config.settings import Settings
    settings = Settings()
    assert settings.project_name == "TARA"


def test_chunker_basic():
    """TelecomChunker should work on simple text."""
    from src.ingestion.chunker import TelecomChunker

    chunker = TelecomChunker(chunk_size=100)
    chunks = chunker.chunk_document(
        "5.3.3 RRC connection establishment\n\nThe purpose of this procedure is to establish an RRC connection.",
        metadata={"filename": "test.pdf", "spec_number": "38.331"},
    )
    assert len(chunks) >= 1
    assert chunks[0].spec_number == "38.331"


def test_intent_classifier():
    """IntentClassifier should route queries correctly."""
    from src.query.intent_classifier import IntentClassifier, QueryIntent

    classifier = IntentClassifier()
    assert classifier.classify("What is PDCCH?") == QueryIntent.QNA
    assert classifier.classify("Root cause of high packet loss") == QueryIntent.RCA
    assert classifier.classify("Compare LTE and 5G NR") == QueryIntent.COMPARISON


def test_entity_extractor():
    """EntityExtractor should find spec refs and KPIs."""
    from src.query.entity_extractor import EntityExtractor

    extractor = EntityExtractor()
    entities = extractor.extract("What does TS 38.331 say about RSRP measurement?")
    assert "TS 38.331" in entities.spec_refs
    assert "rsrp" in entities.kpis


def test_query_router():
    """QueryRouter should produce a RoutedQuery."""
    from src.query.router import QueryRouter, ProcessingPath

    router = QueryRouter()
    routed = router.route("What is the purpose of PDCCH?")
    assert routed.path == ProcessingPath.FAST

    routed_rca = router.route("Root cause analysis: high packet loss on cell 12345 with interference alarms and performance degradation")
    assert routed_rca.path == ProcessingPath.REASONING


def test_telecom_dictionary():
    """Acronym dictionary should have core entries."""
    from src.utils.telecom_dictionary import TELECOM_ACRONYMS

    assert "PDCCH" in TELECOM_ACRONYMS
    assert "NR" in TELECOM_ACRONYMS
    assert len(TELECOM_ACRONYMS) >= 50


def test_rrf_fusion():
    """RRF should combine results correctly."""
    from src.retrieval.hybrid_fusion import HybridFusion
    from src.retrieval.dense_retriever import RetrievedDocument

    fusion = HybridFusion(k=60)

    dense = [RetrievedDocument(id=1, text="doc1", score=0.9),
             RetrievedDocument(id=2, text="doc2", score=0.8)]
    sparse = [RetrievedDocument(id=2, text="doc2", score=0.7),
              RetrievedDocument(id=3, text="doc3", score=0.6)]

    fused = fusion.fuse(dense, sparse, top_k=3)
    assert len(fused) == 3
    # Doc 2 appears in both, so should rank highest
    assert fused[0].id == 2


def test_metrics():
    """Metrics should compute correctly."""
    from src.evaluation.metrics import mean_reciprocal_rank, top_k_accuracy

    results = [[1, 2, 3], [3, 1, 2]]
    truths = [1, 2]

    mrr = mean_reciprocal_rank(results, truths)
    assert mrr > 0

    acc = top_k_accuracy(results, truths, k=3)
    assert acc == 1.0
