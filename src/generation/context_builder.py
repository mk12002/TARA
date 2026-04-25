"""
Context Builder
==================
Builds LLM context from retrieved documents.
"""

from typing import List

from loguru import logger
from src.retrieval.dense_retriever import RetrievedDocument


class ContextBuilder:
    """Builds structured context for LLM from retrieved documents."""

    def __init__(self, max_context_tokens: int = 4000):
        self.max_context_tokens = max_context_tokens

    def build(self, documents: List[RetrievedDocument]) -> str:
        """
        Build a formatted context string from retrieved documents.
        
        Each source is numbered for citation tracking.
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []
        total_tokens = 0

        for i, doc in enumerate(documents, 1):
            source_label = f"[Source {i}]"
            section_info = f" ({doc.section})" if doc.section else ""
            spec_info = f" [{doc.spec_number}]" if doc.spec_number else ""

            header = f"{source_label}{spec_info}{section_info}"
            entry = f"{header}\n{doc.text.strip()}"

            # Rough token estimate
            entry_tokens = int(len(entry.split()) * 1.3)

            if total_tokens + entry_tokens > self.max_context_tokens:
                logger.debug(f"Context truncated at {i-1} sources ({total_tokens} tokens)")
                break

            context_parts.append(entry)
            total_tokens += entry_tokens

        context = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Built context: {len(context_parts)} sources, ~{total_tokens} tokens")
        return context
