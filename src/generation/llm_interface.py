"""
LLM Interface (Ollama)
========================
Abstraction layer for Ollama-based LLM inference.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from config.settings import get_settings


@dataclass
class GenerationConfig:
    """LLM generation parameters."""
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    stop: Optional[List[str]] = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    tokens_used: int = 0
    finish_reason: str = ""


class OllamaClient:
    """Ollama LLM client for local inference."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.settings = get_settings()
        self.base_url = base_url or self.settings.ollama.base_url
        self.model = model or self.settings.ollama.model
        self._client = None

    def _get_client(self):
        """Lazy initialize HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.settings.ollama.timeout,
            )
        return self._client

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> LLMResponse:
        """Generate a response from a prompt."""
        config = config or GenerationConfig()

        response = self._get_client().post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            tokens_used=data.get("eval_count", 0),
            finish_reason=data.get("done_reason", ""),
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Chat-style generation with message history."""
        config = config or GenerationConfig()

        response = self._get_client().post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            tokens_used=data.get("eval_count", 0),
            finish_reason=data.get("done_reason", ""),
        )

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = self._get_client().get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
