"""
Health Check Endpoint
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "TARA"}


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check including dependencies."""
    from src.generation.llm_interface import OllamaClient

    ollama_ok = OllamaClient().is_available()

    return {
        "status": "healthy" if ollama_ok else "degraded",
        "service": "TARA",
        "dependencies": {
            "ollama": "connected" if ollama_ok else "unavailable",
        },
    }
