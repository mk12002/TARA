"""
Qdrant Indexer
===============
Indexes document chunks into Qdrant with both dense and sparse vectors.
"""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.ingestion.chunker import Chunk
from config.settings import get_settings


class Indexer:
    """
    Indexes chunks into Qdrant with:
    - Dense vectors (bge-small-en-v1.5, 384d)
    - Sparse vectors (Qdrant native BM25)
    - Rich payload metadata
    """

    def __init__(
        self,
        embedding_model=None,
        qdrant_client=None,
    ):
        self.settings = get_settings()
        self.embedding_model = embedding_model
        self.client = qdrant_client
        self._initialized = False

    def initialize(self):
        """Lazy initialization of Qdrant client and embedding model."""
        if self._initialized:
            return

        # Initialize Qdrant client (local mode)
        if self.client is None:
            from qdrant_client import QdrantClient
            qdrant_path = self.settings.qdrant.path
            Path(qdrant_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=qdrant_path)
            logger.info(f"Qdrant initialized at: {qdrant_path}")

        # Initialize embedding model
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                self.settings.embedding.model,
                device=self.settings.embedding.device,
            )
            logger.info(f"Embedding model loaded: {self.settings.embedding.model}")

        self._ensure_collection()
        self._initialized = True

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        collection_name = self.settings.qdrant.collection
        collections = [c.name for c in self.client.get_collections().collections]

        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embedding.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.success(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection exists: {collection_name}")

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 32) -> int:
        """
        Index a list of chunks into Qdrant.
        
        Returns:
            Number of chunks indexed
        """
        self.initialize()

        logger.info(f"Indexing {len(chunks)} chunks...")

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]

            # Generate dense embeddings
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Build Qdrant points
            from qdrant_client.models import PointStruct

            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                points.append(PointStruct(
                    id=total + j,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "spec_number": chunk.spec_number,
                        "section": chunk.section,
                        "version": chunk.version,
                        "content_type": chunk.content_type,
                        "token_count": chunk.token_count,
                        **chunk.metadata,
                    },
                ))

            self.client.upsert(
                collection_name=self.settings.qdrant.collection,
                points=points,
            )

            total += len(batch)
            logger.debug(f"Indexed batch {i // batch_size + 1}: {total}/{len(chunks)}")

        logger.success(f"Indexed {total} chunks into Qdrant")
        return total
