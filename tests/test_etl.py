"""Tests for visual embedding extraction and vector loading pipeline."""

import pytest
from src.database.qdrant_client import get_client, create_collection, search_similar
from qdrant_client.models import PointStruct


def test_qdrant_in_memory_connection():
    """Verify in-memory Qdrant client can be created."""
    client = get_client(in_memory=True)
    assert client is not None


def test_collection_creation():
    """Verify a collection is created without error."""
    client = get_client(in_memory=True)
    create_collection(client, name="test_collection", dim=4)
    names = [c.name for c in client.get_collections().collections]
    assert "test_collection" in names


def test_upsert_and_search():
    """Verify that upserted vectors can be retrieved via similarity search."""
    from src.database.qdrant_client import upsert_vectors
    client = get_client(in_memory=True)
    create_collection(client, name="test", dim=4)
    points = [PointStruct(id=i, vector=[float(i)] * 4, payload={"label": f"item_{i}"}) for i in range(5)]
    upsert_vectors(client, points, collection="test")
    results = search_similar(client, query_vector=[1.0, 1.0, 1.0, 1.0], top_k=3, collection="test")
    assert len(results) == 3


def test_malformed_vector_raises():
    """Verify that an empty query vector raises a ValueError."""
    client = get_client(in_memory=True)
    with pytest.raises(ValueError):
        search_similar(client, query_vector=[], top_k=5, collection="test")
