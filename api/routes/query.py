"""
Query Endpoint
=================
Main query endpoint for TARA.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

router = APIRouter()


class QueryRequest(BaseModel):
    """Incoming query request."""
    query: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of sources")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranking")


class SourceInfo(BaseModel):
    """Source document information."""
    source_id: int
    text: str
    spec_number: str = ""
    section: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    """Query response with answer and sources."""
    answer: str
    confidence: str = "MEDIUM"
    sources: List[SourceInfo] = []
    intent: str = "QNA"
    processing_path: str = "FAST"
    citation_count: int = 0


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Process a telecom query through the full TARA pipeline.
    
    Steps: Query Understanding → Retrieval → Generation → Verification
    """
    try:
        # Step 1: Route query
        from src.query.router import QueryRouter
        router_instance = QueryRouter()
        routed = router_instance.route(request.query)

        # Step 2: Retrieve
        from src.retrieval.retrieval_pipeline import RetrievalPipeline
        pipeline = RetrievalPipeline()
        documents = pipeline.retrieve(
            routed.enriched_query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )

        # Step 3: Generate
        from src.generation.generator import Generator
        generator = Generator()
        response = generator.generate(request.query, documents)

        # Step 4: Post-process
        from src.generation.post_processor import PostProcessor
        processor = PostProcessor()
        processed = processor.process(response.text, total_sources=len(documents))

        # Build sources
        sources = [
            SourceInfo(
                source_id=i + 1,
                text=doc.text[:300],
                spec_number=doc.spec_number,
                section=doc.section,
                score=doc.score,
            )
            for i, doc in enumerate(documents)
        ]

        return QueryResponse(
            answer=processed.answer,
            confidence="HIGH" if processed.is_grounded else "LOW",
            sources=sources,
            intent=routed.intent.value,
            processing_path=routed.path.value,
            citation_count=processed.source_count,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
