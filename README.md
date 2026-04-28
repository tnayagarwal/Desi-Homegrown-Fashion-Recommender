# Multimodal Fashion Recommender

A production-grade multimodal recommendation engine for localized fashion catalogs.
Extracts visual embeddings using a pretrained Vision Transformer (ViT-B/32) and stores
them in a Qdrant vector database for sub-millisecond nearest-neighbor retrieval.

## Architecture

```
Image Catalog (raw files)
        |
        v
ETL: extract_visuals.py  (ViT-B/32 → 512-dim embeddings)
        |
        v
ETL: load_vectors.py   (Batch upsert to Qdrant)
        |
        v
  Qdrant Vector DB (self-hosted, persistent volume)
        |
        v
FastAPI /recommend endpoint (cosine similarity search)
```

## Stack
- **Vision:** Hugging Face `transformers` (ViT-B/32)
- **Vector DB:** Qdrant (self-hosted via Docker)
- **API:** FastAPI + Uvicorn
- **Testing:** Pytest with in-memory Qdrant fixtures

## Running Locally

```bash
# 1. Start Qdrant
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ETL pipeline
python -m src.etl.extract_visuals   # Extract embeddings
python -m src.etl.load_vectors      # Load into Qdrant

# 4. Start API server
uvicorn src.api.server:app --reload

# 5. Run tests
pytest tests/ -v
```

> **Note:** Image dataset not included. Provide your own catalog paths in the ETL step.

<!-- Developer comment #7 for optimization and readability check. -->

<!-- Developer comment #10 for optimization and readability check. -->

<!-- Developer comment #11 for optimization and readability check. -->
