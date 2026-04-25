"""
TARA FastAPI Application
==========================
REST API backend for TARA Telecom RAG Assistant.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, health

app = FastAPI(
    title="TARA - Telecom Agentic RAG Assistant",
    description="AI-powered telecom knowledge assistant with citation-backed answers",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, tags=["health"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    pass  # Lazy initialization handles this


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
