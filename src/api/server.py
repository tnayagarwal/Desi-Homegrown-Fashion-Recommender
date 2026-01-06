"""
Fashion Recommender REST API
==============================
FastAPI server exposing recommendation endpoints backed by Qdrant vector search.
"""

import logging
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Fashion Recommender API", version="1.0.0")


class RecommendationRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 10


class RecommendationResponse(BaseModel):
    results: List[dict]


@app.get("/health")
async def health_check():
    """Liveness probe for load balancer and CI pipelines."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Find visually similar fashion items given a query embedding.

    Args:
        request: RecommendationRequest containing embedding and top_k.

    Returns:
        Top-k similar items from the catalog.
    """
    if not request.query_vector:
        raise HTTPException(status_code=400, detail="query_vector must not be empty.")
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100.")

    # Lazy import to avoid startup cost if Qdrant is not configured
    from src.database.qdrant_client import get_client, search_similar
    try:
        client = get_client()
        hits = search_similar(client, request.query_vector, top_k=request.top_k)
        return RecommendationResponse(results=[
            {"id": h.id, "score": h.score, "payload": h.payload} for h in hits
        ])
    except Exception as e:
        logger.error("Recommendation error: %s", e)
        raise HTTPException(status_code=500, detail="Internal recommendation error.")
