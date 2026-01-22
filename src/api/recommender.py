"""
Recommendation Helper
======================
Thin wrapper over the Qdrant search client that produces ranked
fashion catalog recommendations from a visual embedding query.
"""

from typing import List, Dict, Any
from src.database.qdrant_client import get_client, search_similar


def find_similar(
    vector: List[float],
    top_k: int = 10,
    collection: str = "fashion_catalog",
    host: str = "localhost",
) -> List[Dict[str, Any]]:
    """
    Find visually similar fashion items from the catalog.

    Args:
        vector: Query embedding vector (512-dim for ViT-B/32).
        top_k: Number of nearest neighbors to return.
        collection: Qdrant collection name to search.
        host: Qdrant server host.

    Returns:
        List of result dicts with keys: id, score, payload.

    Raises:
        ValueError: If the query vector is empty or None.
    """
    if not vector:
        raise ValueError("Query vector must not be empty or None.")

    client = get_client(host=host)
    hits = search_similar(client, query_vector=vector, top_k=top_k, collection=collection)

    return [
        {"id": h.id, "score": round(h.score, 4), "payload": h.payload}
        for h in hits
    ]