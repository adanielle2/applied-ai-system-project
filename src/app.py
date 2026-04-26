"""
app.py — FastAPI web server for the RAG Music Recommender.

Endpoints:
    POST /recommend    → Get AI-powered song recommendations
    GET  /songs        → List all songs in the catalog
    GET  /health       → Health check
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag_engine import RAGEngine

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
logger.addHandler(_ch)

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Music Recommender",
    description="AI-powered music recommendations using Retrieval-Augmented Generation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG engine at startup
rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
def startup():
    """Initialize the RAG engine when the server starts."""
    global rag_engine
    logger.info("Starting RAG Music Recommender server...")
    try:
        rag_engine = RAGEngine(
            songs_csv_path="data/songs.csv",
            descriptions_path="knowledge_base/song_descriptions.json",
        )
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise


# ── Request / Response Models ──────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """Request body for the /recommend endpoint."""
    genre: str = Field(default="", description="Preferred genre (e.g., pop, lofi, rock)")
    mood: str = Field(default="", description="Preferred mood (e.g., happy, chill, intense)")
    energy: float = Field(default=0.5, ge=0.0, le=1.0, description="Target energy level (0.0–1.0)")
    likes_acoustic: Optional[bool] = Field(default=None, description="Prefer acoustic sounds?")
    free_text: str = Field(default="", description="Free-text description (e.g., 'something for a rainy day')")
    k: int = Field(default=5, ge=1, le=18, description="Number of recommendations to return")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "genre": "lofi",
                    "mood": "chill",
                    "energy": 0.35,
                    "likes_acoustic": True,
                    "free_text": "something calm for studying late at night",
                    "k": 5,
                }
            ]
        }
    }


class SongResponse(BaseModel):
    rank: int
    song_id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    acousticness: float
    explanation: str


class RecommendResponse(BaseModel):
    query: str
    method: str
    recommendations: list[SongResponse]
    retrieved_count: int
    latency_ms: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Get personalized song recommendations using RAG.

    The system:
    1. Converts your preferences into a semantic query
    2. Retrieves the most relevant songs from the knowledge base
    3. Uses an LLM to rank and explain why each song fits you
       (falls back to algorithmic scoring if no LLM is configured)
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    start_time = time.time()

    user_prefs = {
        "genre": req.genre,
        "mood": req.mood,
        "energy": req.energy,
        "likes_acoustic": req.likes_acoustic,
    }

    try:
        result = rag_engine.recommend(
            user_prefs=user_prefs,
            free_text=req.free_text,
            k=req.k,
        )
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"Request completed in {latency_ms}ms")

    return RecommendResponse(
        query=result["query"],
        method=result["method"],
        recommendations=[SongResponse(**r) for r in result["recommendations"]],
        retrieved_count=result["retrieved_count"],
        latency_ms=latency_ms,
    )


@app.get("/songs")
def list_songs():
    """List all songs in the catalog."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return {"songs": rag_engine.songs, "count": len(rag_engine.songs)}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_engine": rag_engine is not None,
        "llm_available": rag_engine.llm_client is not None if rag_engine else False,
        "songs_loaded": len(rag_engine.songs) if rag_engine else 0,
    }
