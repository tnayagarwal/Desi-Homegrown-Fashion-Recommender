"""
Qdrant Vector Database Client
===============================
Handles collection management and vector upsert/search operations
for the multimodal fashion recommendation engine.
"""

import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

# Default vector dimensionality (ViT-B/32 outputs 512-dim embeddings)
VECTOR_DIM = 512
COLLECTION_NAME = "fashion_catalog"


def get_client(host: str = "localhost", port: int = 6333, in_memory: bool = False) -> QdrantClient:
    """
    Return a Qdrant client connected to the specified host.

    Args:
        host: Qdrant server hostname.
        port: Qdrant server port.
        in_memory: If True, use in-memory mode (useful for testing).
    """
    if in_memory:
        return QdrantClient(":memory:")
    return QdrantClient(host=host, port=port)


def create_collection(client: QdrantClient, name: str = COLLECTION_NAME, dim: int = VECTOR_DIM) -> None:
    """
    Create a Qdrant collection if it does not already exist.

    Args:
        client: Active Qdrant client instance.
        name: Collection name.
        dim: Vector dimensionality.
    """
    try:
        existing = [c.name for c in client.get_collections().collections]
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Collection '%s' created with dim=%d.", name, dim)
        else:
            logger.info("Collection '%s' already exists.", name)
    except Exception as e:
        logger.error("Failed to create collection '%s': %s", name, e)
        raise


def upsert_vectors(client: QdrantClient, points: List[PointStruct], collection: str = COLLECTION_NAME) -> None:
    """
    Batch-upsert embedding vectors into the collection.

    Args:
        client: Active Qdrant client instance.
        points: List of PointStruct objects (id, vector, payload).
        collection: Target Qdrant collection name.
    """
    if not points:
        logger.warning("upsert_vectors called with empty points list.")
        return
    client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %d vectors into '%s'.", len(points), collection)


def search_similar(
    client: QdrantClient,
    query_vector: List[float],
    top_k: int = 10,
    collection: str = COLLECTION_NAME,
) -> list:
    """
    Find the top-k nearest neighbors for a given query embedding.

    Args:
        client: Active Qdrant client instance.
        query_vector: The embedding to search with.
        top_k: Number of results to return.
        collection: Target Qdrant collection name.

    Returns:
        List of ScoredPoint results.
    """
    if not query_vector:
        raise ValueError("query_vector must not be empty.")
    return client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
