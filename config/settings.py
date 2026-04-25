"""
TARA Global Settings
=====================
Centralized configuration using Pydantic Settings.
All values can be overridden via environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama LLM inference settings."""
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout: int = 120


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cuda"
    batch_size: int = 32
    dimension: int = 384  # bge-small-en-v1.5 output dimension


class RerankerSettings(BaseSettings):
    """Cross-encoder reranker settings."""
    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda"
    top_k: int = 10


class QdrantSettings(BaseSettings):
    """Qdrant vector database settings (local mode)."""
    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    path: str = "./data/indices/qdrant"
    collection: str = "tara_telecom"


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline settings."""

    dense_top_k: int = 50
    sparse_top_k: int = 50
    rerank_top_k: int = 10
    rrf_k: int = 60  # RRF constant


class GenerationSettings(BaseSettings):
    """LLM generation settings."""

    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9


class Settings(BaseSettings):
    """Master settings container."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project
    project_name: str = "TARA"
    log_level: str = "INFO"
    environment: str = "development"

    # Sub-settings
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)

    # Paths
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    evaluation_data_dir: Path = Path("./data/evaluation")
    cache_dir: Path = Path("./data/cache")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
