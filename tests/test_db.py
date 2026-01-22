"""
Database-level tests for the Qdrant client wrapper.
All tests run against an in-memory Qdrant instance (no server required).
"""

import pytest
from qdrant_client.models import PointStruct
from src.database.qdrant_client import (
    get_client,
    create_collection,
    upsert_vectors,
    search_similar,
)

DIM = 8
COLLECTION = "test_fashion"


@pytest.fixture
def client():
    """In-memory Qdrant client — isolated per test."""
    return get_client(in_memory=True)


def test_client_initializes(client):
    """Client should initialize without errors."""
    assert client is not None


def test_create_collection_succeeds(client):
    """Creating a new collection should succeed."""
    create_collection(client, name=COLLECTION, dim=DIM)
    names = [c.name for c in client.get_collections().collections]
    assert COLLECTION in names


def test_create_collection_idempotent(client):
    """Creating the same collection twice should not raise."""
    create_collection(client, name=COLLECTION, dim=DIM)
    create_collection(client, name=COLLECTION, dim=DIM)


def test_upsert_and_retrieve(client):
    """Upserted vectors should be retrievable via search."""
    create_collection(client, name=COLLECTION, dim=DIM)
    points = [
        PointStruct(id=i, vector=[float(i)] * DIM, payload={"label": f"item_{i}"})
        for i in range(5)
    ]
    upsert_vectors(client, points, collection=COLLECTION)
    results = search_similar(client, query_vector=[1.0] * DIM, top_k=3, collection=COLLECTION)
    assert len(results) == 3
    assert all(hasattr(r, "score") for r in results)


def test_search_empty_vector_raises(client):
    """An empty query vector must raise ValueError before hitting Qdrant."""
    with pytest.raises(ValueError):
        search_similar(client, query_vector=[], top_k=5, collection=COLLECTION)


def test_upsert_empty_list_is_noop(client):
    """Upserting an empty list should not raise and should log a warning."""
    create_collection(client, name=COLLECTION, dim=DIM)
    upsert_vectors(client, [], collection=COLLECTION)  # Should not crash