"""
Fashion Recommender REST API — v1.2.0
=======================================
FastAPI service for multimodal fashion recommendations.
Full OpenAPI 3.0 schema available at /docs and /openapi.json.
"""
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fashion Recommender API",
    description=(
        "Multimodal recommendation engine using Qdrant vector search. "
        "Submit a 512-dim ViT-B/32 embedding to retrieve visually similar "
        "fashion items from the indexed catalog."
    ),
    version="1.2.0",
    contact={"name": "Tanay Agarwal", "url": "https://github.com/tnayagarwal"},
    license_info={"name": "MIT"},
)


class RecommendationRequest(BaseModel):
    query_vector: List[float] = Field(
        ..., description="512-dimensional ViT-B/32 visual embedding.", example=[0.01] * 10
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results.", example=10)
    collection: str = Field(default="fashion_catalog", description="Qdrant collection name.")


class RecommendationItem(BaseModel):
    id: int = Field(..., description="Catalog item ID.")
    score: float = Field(..., description="Cosine similarity score (0-1).")
    payload: dict = Field(default={}, description="Item metadata.")


class RecommendationResponse(BaseModel):
    results: List[RecommendationItem]
    total: int


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe for CI/CD pipelines and load balancers."""
    return {"status": "ok", "service": "fashion-recommender", "version": "1.2.0"}


@app.get("/collections", tags=["Catalog"])
async def list_collections():
    """List all available Qdrant vector collections."""
    try:
        from src.database.qdrant_client import get_client
        client = get_client()
        collections = [c.name for c in client.get_collections().collections]
        return {"collections": collections, "count": len(collections)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {e}")


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(request: RecommendationRequest):
    """
    Find the most visually similar fashion items for a query embedding.

    Submit a 512-dim ViT-B/32 embedding to get back the top-k nearest
    catalog items ranked by cosine similarity.
    """
    from src.database.qdrant_client import get_client, search_similar
    try:
        client = get_client()
        hits = search_similar(client, request.query_vector,
                              top_k=request.top_k, collection=request.collection)
        items = [RecommendationItem(id=h.id, score=round(h.score, 4),
                                   payload=h.payload or {}) for h in hits]
        return RecommendationResponse(results=items, total=len(items))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Recommendation error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error.")

# Developer comment #1 for optimization and readability check.

# Developer comment #2 for optimization and readability check.

# Developer comment #4 for optimization and readability check.

# Developer comment #5 for optimization and readability check.

# Developer comment #6 for optimization and readability check.
