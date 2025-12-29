"""
ETL Step 2 - Vector Loading
=============================
Transforms extracted image embeddings into Qdrant PointStructs
and loads them into the vector database in batches.
"""

import logging
from typing import List, Tuple
from qdrant_client.models import PointStruct
from src.database.qdrant_client import get_client, create_collection, upsert_vectors

logger = logging.getLogger(__name__)
BATCH_SIZE = 256


def run_pipeline(
    embeddings: List[Tuple[str, List[float]]],
    collection: str = "fashion_catalog",
    host: str = "localhost",
) -> None:
    """
    Full ETL load step: connect to Qdrant, create collection, upsert all vectors.

    Args:
        embeddings: List of (image_path, embedding_vector) from extract step.
        collection: Target Qdrant collection name.
        host: Qdrant server hostname.
    """
    client = get_client(host=host)
    create_collection(client, name=collection, dim=len(embeddings[0][1]))

    points = []
    for idx, (path, vector) in enumerate(embeddings):
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={"image_path": path},
        ))

    # Batch upsert to avoid memory overflow on large catalogs
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        upsert_vectors(client, batch, collection=collection)

    logger.info("ETL complete. Loaded %d vectors into '%s'.", len(points), collection)
