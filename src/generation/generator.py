"""
Response Generator
=====================
Generates LLM responses using retrieved context and structured prompts.
"""

from typing import Dict, Optional

import yaml
from loguru import logger
from pathlib import Path

from src.generation.llm_interface import OllamaClient, GenerationConfig, LLMResponse
from src.generation.context_builder import ContextBuilder
from src.retrieval.dense_retriever import RetrievedDocument


class Generator:
    """Generates grounded responses using RAG context."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm = llm_client or OllamaClient()
        self.context_builder = ContextBuilder()
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
        """Load prompt templates from YAML."""
        prompt_path = Path("config/prompts.yaml")
        if prompt_path.exists():
            with open(prompt_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def generate(
        self,
        query: str,
        documents: list,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Generate a grounded response.
        
        Args:
            query: User query
            documents: Retrieved documents
            config: Optional generation config
            
        Returns:
            LLMResponse with generated text
        """
        # Build context from retrieved documents
        context = self.context_builder.build(documents)

        # Format prompt
        prompt_template = self.prompts.get("rag_answer", "{context}\n\nQuestion: {query}\n\nAnswer:")
        prompt = prompt_template.format(context=context, query=query)

        # Generate
        logger.info(f"Generating response for: {query[:60]}...")
        response = self.llm.generate(prompt, config)

        logger.info(f"Generated {response.tokens_used} tokens")
        return response
